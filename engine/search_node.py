"""SearchNode: solution tree node (code, execution, evaluation, search metadata)."""

import copy
import difflib
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from dataclasses_json import DataClassJsonMixin
from engine.executor import ExecutionResult
from config import SearchConfig
from utils.metric import MetricValue
from utils.metric import WorstMetricValue
from utils.response import trim_long_string

logger = logging.getLogger("MLEvolve")


@dataclass(eq=False)
class SearchNode(DataClassJsonMixin):
    """Solution tree node: code, execution results, evaluation, and search metadata."""

    # ---- code & plan ----
    code: str # 该节点对应的代码字符串
    plan: str = field(default=None, kw_only=True)  # type: ignore 生成该代码的计划/思路 
    prompt_input: str | None = field(default=None, kw_only=True)  # type: ignore  输入给 LLM 的提示词

    # ---- general attrs ----
    # 同时跑3个节点，跑完一个补一个，直到累计跑满500个节点为止。搜索树的宽度/深度由 MCTS 的节点选择策略决定，500 只是总预算。
    step: int = field(default=None, kw_only=True)  # type: ignore  当前节点在整个搜索过程中的步骤编号（从0开始递增）STEP_LIMIT=500 → cfg.agent.steps → total_steps，是整个搜索树的节点总数上限，即 journal 里最多生成 500 个节点。 一棵树最多500节点
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["SearchNode"] = field(default=None, kw_only=True)
    children: set["SearchNode"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore  终端输出内容
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True) # 异常信息（如果代码报错)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    analysis: str = field(default=None, kw_only=True)  # type: ignore  LLM 对代码的分析文本
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore  包含数值和优化方向的评估指标
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore 代码是否有 bug / 是否有效
    is_valid: bool = field(default=None, kw_only=True)  # type: ignore

    # ---- search / MCTS ----
    # 每个节点代表一个"代码版本"，通过 MCTS + 贝叶斯采样决定哪个节点值得继续展开，最终找到最优代码。
    stage: Literal["root", "improve", "debug", "draft", "fusion_draft", "evolution", "fusion"]
    visits: int = field(default=0, kw_only=True) # MCTS 标准字段（访问次数、累计奖励、UCT 值）
    total_reward: float = field(default=0.0, kw_only=True) # 累计奖励（UCT 公式中的 Q 的分子）。每次从该节点反向传播时累加子节点的 metric 值，用于计算平均奖励 Q = total_reward / visits。
    is_terminal: bool = field(default=False, kw_only=True) # 是否为终止节点  为 True 时 MCTS 不再展开该节点的子节点（例如达到最大调试深度、或已找到理想解）。
    _uct: float = field(default=0.0, kw_only=True) # 缓存的 UCT 值（前缀 _ 表示内部使用）。uct_value() 方法计算后写入此字段做缓存，避免重复计算
    local_best_node: Optional["SearchNode"] = field(default=None, kw_only=True) # 以本节点为根的子树中当前最优节点的引用。用于剪枝和快速定位局部最优解，避免每次遍历整棵子树。
    is_debug_success: bool = field(default=False, kw_only=True) # 调试是否成功。当一个 debug 阶段节点修复了父节点的 bug 后置为 True，用于统计调试成功率和指导后续策略
    continue_improve: bool = field(default=False, kw_only=True) # 是否继续改进。标记该节点还有改进潜力，即使当前 metric 不是最高，搜索策略也应继续从此节点展开 improve 子节点。
    improve_failure_depth: int = field(default=0, kw_only=True) # 连续改进失败的深度计数。若 improve 子节点连续多次 metric 未提升，此值递增，可用于决定何时放弃继续改进当前分支。
    lock: bool = field(default=False, kw_only=True) # 节点软锁。并发场景下标记该节点"已被某个 worker 认领"，防止多个线程同时对同一节点展开子节点（逻辑锁，非系统互斥锁）。
    child_count_lock: bool = threading.Lock() #  线程互斥锁 用于保护 expected_child_count 的并发读写，见 add_expected_child_count() 和 reached_child_limit()
    expected_child_count: int = field(default=0, kw_only=True) # 预期子节点总数（包含"飞行中"尚未生成完毕的子节点）。并发展开时，worker 启动前先 +1（add_expected_child_count），完成后 -1（sub_expected_child_count），配合 reached_child_limit() 防止超额生成子节点。
    finish_time: str = field(default=None, kw_only=True)  # 节点完成时间戳（字符串格式）。记录该节点代码执行+评估完毕的时刻，用于日志、性能分析和可视化。
    created_time: str = field(default=None, kw_only=True) # 节点创建时间戳

    # ---- Bayesian sampling ----
    alpha: int = field(default=1, kw_only=True) # alpha 和 beta 是 Beta 分布的两个超参数，用于贝叶斯 Thompson 采样  均值p_mean() = alpha / (alpha + beta) 代表该节点历史成功率的贝叶斯估计  每次子节点执行成功 → alpha += 1  每次子节点执行失败 → beta += 1  
    beta: int = field(default=1, kw_only=True) # 相比纯 UCT，Thompson 采样在探索早期（数据少）时更鲁棒，避免过早锁定某个看似优秀但样本不足的节点。 节点被选中的概率正比于从 Beta(alpha, beta) 采样得到的值

    # ---- branch management ----
    branch_id: Optional[int] = field(default=None, kw_only=True) # 搜索分支的唯一编号。整个搜索树被划分为多条"分支"，每条分支是一条从某个节点出发的独立探索路径。
    from_topk: bool = field(default=False, kw_only=True) # 标记该节点是否来自 Top-K 主动利用（exploitation）触发。
    code_summary: Optional[str] = field(default=None, kw_only=True) # LLM 生成的代码方法摘要文本，是全局记忆（Global Memory）系统的核心数据来源。
    work_dir: Optional[str] = field(default=None, kw_only=True) # 节点代码执行时使用的工作目录路径。设计意图是支持每个节点在独立的沙箱目录中运行，避免不同节点的文件产物互相干扰。 但在整个项目中没有被任何代码读取或赋值，属于预留字段（占位符）

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)
        if self.stage not in ["root", "improve", "debug", "draft", "fusion_draft", "evolution", "fusion"]:
            raise ValueError(f"Invalid stage: {self.stage}")

    # ---- base node properties ----

    @property
    def stage_name(self) -> str:
        """Inferred stage based on parent relationship."""
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __eq__(self, other):
        return isinstance(other, SearchNode) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore

    # ---- search methods ----

    
    def update_beta(self, success: bool):
        if success: 
            self.alpha += 1
        else:
            self.beta += 1
            
    def p_mean(self):
        return self.alpha / (self.alpha + self.beta)
    
    
    def uct_value(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate the UCT (Upper Confidence Bound for Trees) value of the current node.
        UCT = Q + c * sqrt(ln(N) / n), where:
        - Q = total_reward / visits (average reward)
        - c = exploration_constant (exploration constant, default is sqrt(2))
        - N = parent_visits (number of visits to the parent node)
        - n = visits (number of visits to the current node)
        """
        parent_visits: int | None = None
        if self.parent:
            parent_visits = self.parent.visits
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have the highest priority
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * (math.log(parent_visits) / self.visits) ** 0.5
        self._uct = exploitation + exploration
        return self._uct

    def reached_child_limit(self, scfg: SearchConfig, for_topk: bool = False) -> bool:
        """Whether this node has reached its child limit (draft/improve/debug). for_topk uses higher limit."""
        # 判断当前节点是否已达到子节点上限，返回 True 表示"不能再展开新子节点"。
        with self.child_count_lock: # 加互斥锁，保护后续对 children、expected_child_count 的读取，防止多线程并发时数据竞争。
            if self.step == 0: # 根节点，其子节点都是初始草稿
                regular_draft_count = sum(1 for child in self.children if child.stage == "draft") # 统计已完成生成的普通草稿子节点数（排除 fusion_draft 等其他类型）
                # expected_child_count includes in-flight children; estimate in-flight drafts
                in_flight = max(0, self.expected_child_count - len(self.children)) # 计算飞行中（in-flight） 的子节点数：expected_child_count 是"已预订但还没生成完"的总预期数，减去已经存在的子节点数，就是还在生成中的数量。
                regular_expected = regular_draft_count + in_flight # 有效草稿总数 = 已完成的草稿 + 正在生成的草稿，代表"最终会存在的草稿数量"
                logger.info(f"[reached_child_limit] node {self.id} regular_draft_count={regular_draft_count}, in_flight={in_flight}, limit={scfg.num_drafts}")
                return regular_expected >= scfg.num_drafts # 若有效草稿总数 ≥ 配置的草稿上限 num_drafts，则返回 True（拒绝再生成新草稿）。
            else: # 非根节点（step > 0）
                if self.is_buggy: # 当前节点有 bug → 展开 debug 子节点
                    if self.has_no_bug_child(): # 如果子节点中已有至少一个无 bug 的节点，立即返回 True，不再继续调试。 逻辑：只要修好过一次，这条调试路径就算完成，无需再生成更多 debug 节点。
                        return True
                    else:
                        return self.expected_child_count >= scfg.num_bugs # 如果还没修好，则检查已预期的调试子节点数是否达到 num_bugs 上限（防止无限重试）。
                else: # 当前节点无 bug → 展开 improve 子节点
                    if for_topk: # 调用方是否来自 Top-K 利用模式（允许更高上限）
                        topk_max_improves = getattr(scfg, 'topk_max_improves', 10) # Top-K 利用模式：使用更宽松的上限 topk_max_improves（默认 10，比普通 num_improves 更大），允许对高质量节点做更多次改进尝试。
                        return self.expected_child_count >= topk_max_improves
                    else:
                        regular_expected = sum(
                            1 for child in self.children
                            if not getattr(child, 'from_topk', False)
                        ) # 普通探索模式：只统计 from_topk=False 的子节点，即排除 Top-K 触发的子节点。这样 Top-K 额外生成的节点不会占用普通 improve 的配额，两套预算互相独立。
                        regular_expected += (self.expected_child_count - len(self.children)) # 加上飞行中的子节点数（与根节点处理飞行中草稿的逻辑相同），估算最终会有多少普通 improve 子节点。  飞行中的节点无法区分是否 top-k，全部计入
                        # 这个误差会导致 regular_expected 偏大，使 reached_child_limit() 更容易提前返回 True，即：
                        # 本来还有配额生成普通 improve 子节点，但因为飞行中的 top-k 节点被误算进来，导致判断为"已满"，错误地拦截了新的普通 improve 展开。
                        return regular_expected >= scfg.num_improves # 若普通 improve 子节点总数 ≥ num_improves 上限，返回 True。 

    
    def update(self, result, add=True):
        if add:
            self.visits += 1
            self.total_reward += result
        
    def has_no_bug_child(self):
        for child in self.children:
            if not child.is_buggy:
                return True
        return False

    @property
    def num_children(self):
        return len(self.children)

    def fetch_child_memory(self, include_code=False):
        """Build memory string from children for the model (include draft nodes; optionally include code diff)."""
        # 核心作用：将当前节点的所有子节点的执行结果整理成一段结构化文本，作为"记忆"注入给 LLM，让它知道"之前试过哪些方案、结果如何"，从而生成更好的下一版代码。
        logger.info("fetch_child_memory")
        summary = []

        sorted_children = sorted(
            [n for n in self.children if n.is_buggy is not None or n.stage == "draft"], # 过滤条件：保留已执行（is_buggy is not None）或草稿阶段（stage == "draft"）的子节点，排除其他中间状态。
            key=lambda n: (
                n.is_buggy is False,
                n.is_buggy is not None,
                # n.metric.value if (n.metric and n.metric.value is not None) else float('-inf')
                n.metric if n.metric is not None else WorstMetricValue()
            ),
            reverse=True
        ) # 无 bug + metric 最高: 最成功的方案，LLM 最应参考;  无 bug + metric 较低:成功但一般;  有 bug（已执行）: 失败方案，提示 LLM 避开   待执行（is_buggy=None）: 草稿，还不知道好不好 

        for idx, n in enumerate(sorted_children, 1):
            summary_part = f"Attempt #{idx}:\n"
            summary_part += f"Design: {n.plan}\n"

            if include_code and self.code and n.code:
                code_diff = self._compute_code_diff(self.code, n.code)
                if code_diff:
                    summary_part += f"Code Changes:\n{code_diff}\n"
                else:
                    summary_part += f"Code Changes: (minimal or formatting changes only)\n"

            if n.is_buggy is None: # 草稿
                summary_part += f"Status: Code generated, execution pending (will run in parallel with other drafts).\n"
            elif n.is_buggy is True:
                summary_part += f"Results: The implementation of this design has bugs.\n"
                summary_part += f"Insight: Using a different approach may not result in the same bugs as the above approach.\n"
            else:
                if n.analysis:
                    summary_part += f"Results: {n.analysis}\n"
                if n.metric and n.metric.value is not None:
                    metric_display = self._format_metric_change(n)
                    summary_part += f"Validation Metric: {metric_display}\n"
                if hasattr(n, 'exec_time') and n.exec_time is not None:
                    summary_part += f"Execution Time: {n.exec_time:.2f}s\n"

            summary.append(summary_part)

        if len(summary) == 0:
            summary.append("")
        else:
            total_attempts = len(sorted_children)
            pending = [n for n in sorted_children if n.is_buggy is None]
            executed = [n for n in sorted_children if n.is_buggy is not None]
            successful = [n for n in executed if n.is_buggy is False]

            stats_parts = []
            if pending:
                stats_parts.append(f"{len(pending)} pending execution")
            if executed:
                stats_parts.append(f"{len(executed)} executed")
                if successful:
                    # best_metric = max(n.metric.value for n in successful if n.metric and n.metric.value is not None)
                    best_metric = max(n.metric for n in successful if n.metric and n.metric.value is not None)
                    stats_parts.append(f"{len(successful)} successful (best: {best_metric:.4f})")
                else:
                    stats_parts.append(f"0 successful (all failed or buggy)")

            stats = f"Summary: {total_attempts} total attempts - " + ", ".join(stats_parts)
            summary.insert(0, stats + "\n")

        return "\n-------------------------------\n".join(summary)

    def _format_metric_change(self, node) -> str:
        """Format metric change for display (respects maximize/minimize)."""
        if not node.metric or node.metric.value is None:
            return "N/A"

        current_val = node.metric.value

        if (node.parent and
            hasattr(node.parent, 'is_buggy') and
            node.parent.is_buggy is False and
            node.parent.metric and
            node.parent.metric.value is not None):

            parent_val = node.parent.metric.value
            raw_change = current_val - parent_val

            if hasattr(node.metric, 'maximize'):
                if node.metric.maximize:
                    improvement = raw_change
                    direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                else:
                    improvement = -raw_change
                    direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            else:
                improvement = raw_change
                direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"

            return f"{parent_val:.4f} → {current_val:.4f} ({improvement:+.4f} {direction})"
        else:
            return f"{current_val:.4f}"

    def _compute_code_diff(self, parent_code: str, child_code: str, context_lines: int = 3) -> str:
        """Compute formatted diff between parent and child code."""
        parent_lines = parent_code.splitlines(keepends=True)
        child_lines = child_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            parent_lines,
            child_lines,
            fromfile='Parent Code', 
            tofile='Modified Code',
            lineterm='', # 不额外添加换行符（已由 keepends=True 保留）
            n=context_lines  # 每处变动前后各保留 3 行上下文
        )

        diff_lines = list(diff)
        if not diff_lines:
            return ""

        formatted_diff = []
        for line in diff_lines[2:]:  #  跳过 "--- Parent Code" 和 "+++ Modified Code" 两行文件头
            if line.startswith('@@'): # 不显示（去掉行号信息）
                continue
            elif line.startswith('+') and not line.startswith('+++'): # 新增行
                formatted_diff.append(f"  + {line[1:]}")
            elif line.startswith('-') and not line.startswith('---'): # 删除行
                formatted_diff.append(f"  - {line[1:]}") 
            elif not line.startswith(('---', '+++')):  # 其他（上下文行）	保留，但限制总行数 < 100	内容（4空格缩进）
                if len(formatted_diff) < 100:
                    formatted_diff.append(f"    {line}")

        if len(formatted_diff) > 100:
            formatted_diff = formatted_diff[:100]
            formatted_diff.append("  ... (diff truncated, too many changes)")

        return '\n'.join(formatted_diff) if formatted_diff else ""

    def fetch_parent_memory(self, include_code=False):
        logger.info("fetch_parent_memory")
        summary = []
        if self.parent is not None and self.parent.is_buggy is not None and self.parent.is_buggy is False:
            summary_part = f"Design: {self.parent.plan}\n"
            if include_code:
                summary_part += f"Code: {self.parent.code}\n"
            summary_part += f"Results: {self.parent.analysis}\n"
            summary_part += f"Validation Metric: {self.parent.metric.value}\n"
            if hasattr(self.parent, 'exec_time') and self.parent.exec_time is not None:
                summary_part += f"Execution Time: {self.parent.exec_time:.2f}s\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)
    
    def add_expected_child_count(self): # 两个方法配对使用，管理"预订计数器"的生命周期
        with self.child_count_lock:
            self.expected_child_count += 1
            logger.info(f"current {self.id} expected_child_count is {self.expected_child_count}.")
            
            
    def sub_expected_child_count(self):
        with self.child_count_lock:
            self.expected_child_count -= 1
            logger.info(f"current {self.id} expected_child_count is {self.expected_child_count}.")

    def __getstate__(self): # 这两个是 Python pickle 序列化协议的钩子，用于控制对象的序列化和反序列化行为。
        state = self.__dict__.copy()
        state.pop('child_count_lock', None) 
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.child_count_lock = threading.Lock()
    
    def generate_node_trajectory(self, need_code=False) -> str:
        """Return formatted trajectory string for this node."""
        summary_part = f""
        if hasattr(self, 'branch_id') and self.branch_id:
            summary_part += f"Branch ID: {self.branch_id}\n"

        summary_part += f"Stage: {self.stage.upper()}\n"
        if self.plan:
            summary_part += f"Design: {self.plan}\n"

        if self.code and need_code:
            summary_part += f"Code: {self.code}\n"

        if self.is_buggy is True:
            summary_part += f"Results: The implementation of this design has bugs.\n"
            if self.analysis:
                summary_part += f"Analysis: {self.analysis}\n"
        elif self.is_buggy is False:
            if self.analysis:
                summary_part += f"Results: {self.analysis}\n"
            if self.metric and self.metric.value is not None:
                metric_display = self._format_metric_change(self)
                summary_part += f"Validation Metric: {metric_display}\n"
            if hasattr(self, 'exec_time') and self.exec_time is not None:
                summary_part += f"Execution Time: {self.exec_time:.2f}s\n"

        else:
            summary_part += f"Results: Step not yet executed.\n"
            logger.warning(f"Node {self.id} is not executed.")
        
        return summary_part
    
    def get_root_to_current_trajectory(self, max_steps: int = None, llm_summary_threshold: int = 5) -> str:
        """Return formatted trajectory from root to this node (optionally limited to max_steps)."""
        trajectory = self._get_trajectory_raw(max_steps)
        return self._get_trajectory_full(trajectory)
    
    def _get_trajectory_raw(self, max_steps: int = None) -> List[str]:
        """Collect raw trajectory steps from this node up to root."""
        trajectory = []
        current = self
        while current and current.parent:
            step_trajectory = current.generate_node_trajectory()
            trajectory.append(step_trajectory)
            current = current.parent
            if max_steps and len(trajectory) >= max_steps:
                break
        return list(reversed(trajectory))
    
    def _get_trajectory_full(self, trajectory: List[str]) -> str:
        """Format trajectory as Step 1: ..., Step 2: ..."""
        trajectory_parts = []
        
        for i, step_trajectory in enumerate(trajectory):
            step_header = f"Step {i+1}:"
            step_info = f"{step_header}\n{step_trajectory}"
            trajectory_parts.append(step_info)
        
        return "\n-------------------------------\n".join(trajectory_parts)
    
    
# ---------------------------------------------------------------------------
# Journal — ordered collection of SearchNodes forming the solution tree
# ---------------------------------------------------------------------------

@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[SearchNode] = field(default_factory=list)

    def __getitem__(self, idx: int) -> SearchNode:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: SearchNode) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[SearchNode]:
        """Return a list of nodes representing initial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def good_nodes(self) -> list[SearchNode]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self, only_good=True) -> None | SearchNode:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return max(nodes, key=lambda n: n.metric)


def get_path_to_node(journal: Journal, node_id: str) -> list[str]:
    path = [node_id]
    node2parent = {n.id: n.parent.id for n in journal.nodes if n.parent is not None}
    while node_id in node2parent:
        parent_id = node2parent[node_id]
        path.append(parent_id)
        node_id = parent_id
    return path[::-1]


def get_longest_path(journal: Journal) -> list[str]:
    longest_path = []
    for node in journal.nodes:
        path = get_path_to_node(journal, node.id)
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path


def filter_on_path(journal: Journal, path: list[str]) -> Journal:
    journal_copy = copy.deepcopy(journal)
    journal_copy.nodes = [n for n in journal_copy.nodes if n.id in path]
    for n in journal_copy.nodes:
        n._term_out = "<OMITTED>"
        n.exc_stack = "<OMITTED>"
    return journal_copy


def filter_for_best_path(journal: Journal, best_node: str) -> Journal:
    path_to_best = get_path_to_node(journal, best_node)
    return filter_on_path(journal, path_to_best)


def filter_for_longest_path(journal: Journal) -> Journal:
    longest_path = get_longest_path(journal)
    return filter_on_path(journal, longest_path)


def filter_journal(journal: Journal) -> Journal:
    best_node = journal.get_best_node(only_good=True)
    if best_node is not None:
        return filter_for_best_path(journal, best_node.id)
    else:
        return filter_for_longest_path(journal)

