"""AgentSearch: tree search coordinator; delegates to node_selection, evaluation, execution, solution_manager."""

import logging
import random
import time
from typing import Callable, List, Dict

from engine.executor import ExecutionResult
from engine.search_node import SearchNode, Journal
import utils.data_preview as data_preview
from config import Config
from utils.metric import WorstMetricValue
import threading
import json

from agents import (
    draft_agent, improve_agent, debug_agent,
    evolution_agent, fusion_agent, aggregation_agent,
    code_review_agent,
    result_parse_agent,
)
from engine import node_selection, evaluation, execution, solution_manager
from engine.conditions import is_branch_stagnant
from utils.data_preview import clean_task_desc

logger = logging.getLogger("MLEvolve")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

class AgentSearch:
    def __init__(
            self,
            task_desc: str,
            cfg: Config,
            journal: Journal,
    ):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        self.task_desc = clean_task_desc(task_desc, cfg)
        self.journal = journal
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: SearchNode | None = None
        self.all_root = True
        self.virtual_root = SearchNode(parent=None, plan="(root)", code="", metric=WorstMetricValue(),
                                     stage="root")
        self.current_node_list = []
        self.journal.append(self.virtual_root)
        self.best_metric: float = None
        self.best_node: SearchNode = None
        self.search_start_time = None
        self.journal_lock = threading.Lock()
        self.save_node_lock = threading.Lock()
        self.start_time = time.time()
        self.use_stepwise_generation = True

        self.next_branch_id = 1
        self.branch_all_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_successful_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_node_count: Dict[int, int] = {}
        self.use_coldstart = cfg.coldstart.use_coldstart
        self.coldstart_description = cfg.coldstart.description

        # Top-N candidates
        self.top_k = self.scfg.top_candidates_size
        self.top_candidates: List[SearchNode] = []

        # Performance stagnation detection
        self.best_metric_history = []
        self.stagnation_threshold = self.scfg.stagnation_window
        self.post_process_triggered = False
        self.post_process_attempts = 0
        self.max_post_process_attempts = 4
        self.improve_attempts_count = 0
        self.last_successful_improve_step = 0

        self.fusion_draft_count = 0
        self.max_fusion_drafts = cfg.agent.max_fusion_drafts

        self.metric_maximize: bool | None = None
        self.metric_maximize_reasoning: str | None = None
        result_parse_agent.determine_metric_direction(self)

        # Global memory
        self.global_memory = None
        if self.acfg.use_global_memory:
            try:
                from agents.memory.global_memory import GlobalMemoryLayer
                memory_dir = str(self.cfg.workspace_dir / "global_memory")
                self.global_memory = GlobalMemoryLayer(
                    memory_dir=memory_dir,
                    embedding_model_path=self.acfg.memory_embedding_model_path,
                    embedding_device=self.acfg.memory_embedding_device,
                    similarity_threshold=self.acfg.memory_similarity_threshold,
                )
                logger.info(f"[AgentSearch] Global memory enabled and initialized at {memory_dir}")
            except Exception as e:
                import traceback
                logger.warning(f"[AgentSearch] Failed to initialize global memory: {e}")
                logger.debug(f"[AgentSearch] Global memory initialization traceback: {traceback.format_exc()}")
                self.global_memory = None
        else:
            logger.info("[AgentSearch] Global memory is disabled by config")

    def _serialize_prompt(self, prompt_complete) -> str | None:
        """Serialize prompt (str or dict) to string for saving in node."""
        if prompt_complete is None:
            return None
        if isinstance(prompt_complete, str):
            return prompt_complete
        elif isinstance(prompt_complete, dict):
            return json.dumps(prompt_complete, ensure_ascii=False, indent=2)
        else:
            return str(prompt_complete)

    def update_data_preview(self):
        base_preview = data_preview.generate(self.cfg.workspace_dir)
        submission_format_warning = """

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        """
        self.data_preview = base_preview + submission_format_warning

    def is_root(self, node: SearchNode):
        return node.id is self.virtual_root.id

    def _run_single_step(self, parent_node: SearchNode, exec_callback: ExecCallbackType, execute_immediately: bool = True):
        """Run one search step: select action (draft/debug/improve), execute, parse, validate.
        整个系统最核心的函数，一次完整的"生成代码 → 执行 → 评估"闭环都在这里完成。
        
        """
        result_node = None
        _root = False
        # is_terminal 在以下情况被置为 True ：连续改进失败次数超过 improve_failure_depth 上限  找到了理想解（metric 达到最优）  
        # Terminal 节点直接 backpropagate reward=0，不再生成子节点。
        if not parent_node.is_terminal: 
            try:
                if self.is_root(parent_node):
                    if parent_node.reached_child_limit(scfg=self.scfg): # 多分支聚合，融合已有最优解
                        logger.info("🎯 Regular draft limit reached, triggering multi-branch aggregation (conditions already checked in select())")
                        result_node = aggregation_agent.run(self, mode="node", parent_node=parent_node) # 多分支聚合，融合已有最优解
                        if result_node:
                            result_node.lock = True
                            logger.info(f"[_run_single_step] Aggregation branch node {result_node.id} is locked.")
                        else:
                            logger.info("Aggregation failed or limit reached, skipping. Will continue normal search.")
                            result_node = None
                    else:
                        result_node = draft_agent.run(self) # 从零生成初始方案
                        result_node.lock = True
                        logger.info(f"[_run_single_step] Draft node {result_node.id} is locked.")
                elif parent_node.is_buggy or parent_node.is_valid is False: # is_buggy=True 表示执行报错；is_valid=False 表示代码结构不合法。
                    result_node = debug_agent.run(self, parent_node)
                
                # 已执行、且没有bug：判断是否来自 Top-K 利用模式，使用不同的停滞阈值
                elif parent_node.is_buggy is False:
                    can_use_fusion = False
                    if self.search_start_time:
                        elapsed_time = time.time() - self.search_start_time
                        if elapsed_time >= self.acfg.time_limit / 2:
                            can_use_fusion = True
                    is_from_topk = getattr(parent_node, '_topk_triggered', False)
                    # 判断是否来自 Top-K 利用模式，使用不同的停滞阈值（停滞理解为连续改进失败/改进幅度较小）
                    # Top-K 节点本身是当前较优解，通常处在“精修区间，提升幅度会更小、更慢，容易出现短期不提升但后面还能继续涨
                    # 如果对 Top-K 也用 3 次就判停滞，会太早切走到别的策略（evolution/fusion），打断 exploitation。
                    # 给 Top-K 更大阈值（6），意思是“允许它多试几次再判死”。
                    # threshold=3：最近 3 次成功都没破 best 就算停滞（更敏感）
                    # threshold=6：要最近 6 次都没破 best 才算停滞（更宽容）
                    stagnation_threshold = self.scfg.topk_stagnation_threshold if is_from_topk else self.scfg.branch_stagnation_threshold
                    if is_from_topk:
                        logger.info(f"🎯 Exploitation mode: using relaxed stagnation threshold ({stagnation_threshold} attempts)")
                    
                    # 取该分支最近 threshold 个成功节点（recent_nodes）
                    # 看这些节点是否都没超过该分支历史最优 branch_best_metric
                    # 如果“最近窗口全没提升”且窗口至少有 2 个节点，就判 stagnant=True
                    if is_branch_stagnant(self, parent_node.branch_id, threshold=stagnation_threshold):
                        if can_use_fusion: # # 已过 50% 时间限制，且分支停滞超过阈值，随机选择融合或进化策略，增加搜索多样性和跳出局部最优的机会。
                            if random.random() < self.acfg.fusion_vs_evolution_prob: 
                                logger.info(f"🎯 Triggering fusion for stagnant node {parent_node.id} (after 6h)")
                                result_node = fusion_agent.run(self, parent_node)   # 跨分支融合
                            else:
                                logger.info(f"🎯 Triggering intra-branch evolution for stagnant node {parent_node.id} (after 6h)")
                                result_node = evolution_agent.run(self, parent_node) # 分支内进化
                        else:
                            logger.info(f"🔄 Using evolution for stagnant node {parent_node.id} (before 6h)")
                            result_node = evolution_agent.run(self, parent_node)  # 时间未到 50% 前，直接使用进化策略改进同一分支的方案，尝试跳出局部最优。
                    else: # 没有经过停滞检测，正常使用 improve 策略改进当前分支方案。
                        logger.info(f"🔄 Using normal improve for node {parent_node.id}")
                        result_node = improve_agent.run(self, parent_node)  # 正常改进

                else:
                    logger.warning(f"[_run_single_step] node {parent_node.id} is_buggy is None.")

                if result_node: # 代码审查和执行放在同一个 if 里，意思是：无论是 draft/debug/improve/evolution/fusion 哪个策略生成了新节点，都要经过代码审查；如果没有生成新节点（result_node=None），就直接跳过审查和执行，回到上层重新选节点。
                    reviewed_code = code_review_agent.run(self, result_node)

                    if reviewed_code.strip() != result_node.code.strip():
                        logger.info(f"Node {result_node.id} code has been reviewed and modified")
                        result_node.code = reviewed_code
                    else:
                        logger.info(f"Node {result_node.id} passed code review without changes")

                    if not execute_immediately: # 如果 execute_immediately=False，表示只生成代码但不执行，
                        logger.info(f"Node {result_node.id} code generated and reviewed, execution deferred")
                        result_node.pending_execution = True
                        return _root, result_node
                    exe_res = exec_callback(result_node.code, result_node.id, True) # 执行新节点代码，获取执行结果（包括 metric、是否报错等信息）
                    result_node = result_parse_agent.run(self,
                        node=result_node,
                        exec_result=exe_res
                    ) # 解析结果，更新节点状态（metric、is_buggy、is_valid 等）
                    execution.validate_executed_node(self, result_node) # TODO 至少要修改提交文件的格式
                    logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

                    if parent_node.is_buggy and result_node.is_buggy is False:
                        parent_node.is_debug_success = True  # 标识父节点的 debug 成功，供后续分析使用

                    _root = evaluation.check_improvement(self, result_node, parent_node) # 评估改进  _root代表是否需要回到根节点重新选分支（终端节点或需要重新选择）。 终端节点 = 这条路已经走到头了，继续展开也没意义。_root = True  →  这个节点触发了反向传播（分支已结束或失败） _root = False →  节点有改进潜力，继续在这条分支上展开
                    with self.journal_lock:
                        if self.best_node and result_node.metric.maximize and self.best_node.metric.maximize != result_node.metric.maximize:
                            logger.warning(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                            raise ValueError(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                        else:
                            self.journal.append(result_node)

            except Exception as e:
                logger.warning(f"Step failed for parent {parent_node.id}, rolling back expected child count and propagating zero reward.")
                evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False) # 对 parent 做 reward=0 的 backpropagate（不加入树）
                parent_node.sub_expected_child_count() # 减少 expected_child_count（修正并发计数）
                raise e

        else:
            evaluation.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node

    def step(self, node: SearchNode, exec_callback: ExecCallbackType, execute_immediately: bool = True) -> SearchNode:
        """
        step 函数有两种调用模式：

        模式 A（并发/外部指定）：外部调用者已经知道要展开哪个节点，直接把它传进来。这在并发搜索中使用，调度器从外部分配任务。
        模式 B（顺序/自动选择）：调用者不知道该扩展哪个节点，传 None 或 virtual_root 进来，意思是"你来决定"。
        virtual_root 是构造函数里创建的一个占位符节点（stage="root"），它不是真正的解决方案节点，只是一个"我没有指定节点"的信号。
        收到这个信号后，select_with_soft_switch 才接管，执行真正的 MCTS 节点选择策略
        
        
        调用方通常会写这样的循环：
        node = agent.virtual_root
        while not done:
            node = agent.step(node, exec_callback)
            # 如果返回 virtual_root → 自动触发 select_with_soft_switch 重新选节点
            # 如果返回新节点       → 下次继续从这个节点展开（同一分支深入）
        
        """
        
        
        # 初始化检查
        if not self.journal.nodes or self.data_preview is None:
            # 生成数据预览（供 Agent 理解数据集）
            self.update_data_preview()
            # 记录搜索开始时间（用于后续判断是否超过半程时间，触发 fusion/evolution 策略）。
            self.search_start_time = time.time()
        
        # 选节点（若需要）
        # 如果传入的是根节点或空节点，则调用 select_with_soft_switch 自动选一个父节点。这支持两种调用方式： 外部指定节点（异步/并发模式）自动选节点（单线程顺序搜索） 
        if not node or node.stage == "root": # 本质是区分"外部指定节点"和"自主选节点"两种工作模式
            node = node_selection.select_with_soft_switch(self)
        
        # 执行核心逻辑：生成/改进代码 → 代码审查 → 执行 → 结果解析 → 评估 → 更新日志和最佳解
        #  会根据父节点状态选择策略 （draft/debug/improve/evolution/fusion），并在生成代码后进行代码审查，最后执行并解析结果。执行可以选择立即执行或延后执行（生成代码但不运行）。
        # root 是布尔标志，表示此步是否"回到根"（终端节点或需要重新选择）。 终端节点 = 这条路已经走到头了，继续展开也没意义。_run_single_step 里检测到 is_terminal 时直接反向传播奖励 0，不再生成子节点
        # _root = True  →  这个节点触发了反向传播（分支已结束或失败） _root = False →  节点有改进潜力，继续在这条分支上展开
        _root, result_node = self._run_single_step(node, exec_callback=exec_callback, execute_immediately=execute_immediately)
        
        # 更新最优解
        if result_node:
            metric_value = result_node.metric.value if result_node.metric else None
            best_metric = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
            logger.info(f"[step] {node.id} → {result_node.id}: metric={metric_value}, best={best_metric}")
        
        # 若新节点有有效 metric，则尝试更新全局最优解（写文件、更新 self.best_node）。
        if result_node and result_node.metric and result_node.metric.value is not None:
            solution_manager.update_best_solution(self, result_node)
        
        # 记录当前步数、总节点数、分支数、当前最优值，便于监控搜索进展。
        self.current_step = len(self.journal)

        # Cumulative stats 
        total_nodes = len(self.journal)
        n_branches = len(self.branch_all_nodes)
        best_val = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
        logger.info(f"[stats] step={self.current_step}, nodes={total_nodes}, branches={n_branches}, best={best_val}")
        
        # virtual_root 不是真的"回到根节点"，而是一个重置信号，告诉搜索引擎：当前分支已终止，请从全局视角重新选一个值得展开的节点。
        if _root or result_node is None:
            return self.virtual_root # 让上层重新选节点，如果返回 virtual_root → 自动触发 select_with_soft_switch 重新选节点 virtual_root 是构造函数里创建的一个占位符节点（stage="root"），它不是真正的解决方案节点，只是一个"没有指定节点"的信号。
        else:
            return result_node # # 返回新生成的节点  返回值决定下一轮 step 的输入，下次继续从这个节点展开（同一分支深入）。返回 virtual_root 意味着"本步没有产出，请重新选节点"；返回 result_node 则可继续在该节点上深入搜索。

    def execute_deferred_node(self, node: SearchNode, exec_callback: ExecCallbackType) -> SearchNode:
        """Execute a node that was generated and reviewed but not yet run (pending_execution=True)."""
        if not hasattr(node, 'pending_execution') or not node.pending_execution:
            logger.warning(f"Node {node.id} is not marked for deferred execution")
            return node

        logger.info(f"Executing deferred node {node.id}")
        parent_node = node.parent

        try:
            exe_res = exec_callback(node.code, node.id, True)
            node = result_parse_agent.run(self,
                node=node,
                exec_result=exe_res
            )

            execution.validate_executed_node(self, node)

            logger.info(f"Node {node.id} execution completed: metric={node.metric.value}, is_buggy={node.is_buggy}")

            node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

            if parent_node and parent_node.is_buggy and node.is_buggy is False:
                parent_node.is_debug_success = True

            _root = evaluation.check_improvement(self, node, parent_node)

            with self.journal_lock:
                if self.best_node and node.metric.maximize and self.best_node.metric.maximize != node.metric.maximize:
                    logger.warning("New node's metric is inconsistent with metrics in the journal")
                    raise ValueError("New node's metric is inconsistent with metrics in the journal")
                else:
                    self.journal.append(node)
                    logger.info(f"Node {node.id} added to journal")

            node.pending_execution = False
            solution_manager.update_best_solution(self, node)

            return node

        except Exception as e:
            logger.exception(f"Exception during deferred node execution: {e}")
            evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
            parent_node.sub_expected_child_count()
            raise e
