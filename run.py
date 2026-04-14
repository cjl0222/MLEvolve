import atexit
import logging
import sys
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from utils.logging_config import setup_logging
import torch



def run():
    cfg = load_cfg() #  读取config.yaml 可命令行覆盖
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)

    #  在agent 开始探索之前，预先注入领域知识（模型列表、代码模板），让 agent 的第一步不是盲目乱猜，而是有方向地出发。
    #  有 coldstart → 用知识库"预热"agent，缩短冷启动阶段的探索时间，提升初始代码质量；没有 coldstart → 直接空手起家，完全依赖 agent 的自我探索能力，可能需要更多的试错才能找到有效的解决方案。
    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base") # 包括模型描述和代码模板等，构建成文本形式的指导信息，供智能体在生成代码时参考
        cfg.coldstart.description = build_guidance_description(cfg)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    with Status("Preparing agent workspace (copying and extracting files) ..."): # 可视化组件 用于在终端显示动态状态指示器
        # 复制和提取必要的文件到工作目录，确保智能体有一个干净、结构化的环境来进行代码生成和执行。这个步骤可能包括：
        # 1. 创建工作目录结构（input、working、submission等子目录）
        # 2. 从数据目录复制输入文件到工作目录的input子目录
        # 3. 如果需要，进行数据预处理（例如解压缩、格式转换等），确保输入数据符合智能体的预期格式和要求。

        prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir) #   若未产生步骤则清理工作目录

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

    lock = threading.Lock()
    completed = 0

    # 配置项：initial_drafts（初始草稿数） 目标：快速生成 N 个不同的代码方案（仅生成，不执行）
    pending_draft_nodes = []
    if initial_draft_count > 0 and total_steps > 0:
        logger.info(f"📝 Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

        def step_task_generate_only():
            logger.info(f"[step_task_generate_only] Generating draft from virtual root")
            return agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)

        for draft_idx in range(min(initial_draft_count, total_steps)):
            try:
                logger.info(f"🔨 Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)")
                cur_node = step_task_generate_only()
                pending_draft_nodes.append(cur_node)
                logger.info(f"✅ Draft {draft_idx + 1} code generated: node.id={cur_node.id}, added to virtual_root.children")

            except Exception as e:
                logger.exception(f"❌ Exception during draft {draft_idx + 1} generation: {e}")

        logger.info(f"✅ Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")

    if pending_draft_nodes or completed < total_steps:
        logger.info(f"🚀 Phase 2: Pipelined parallel execution")
        logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
        logger.info(f"   - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            try:
                executed_node = agent.execute_deferred_node(node, exec_callback)
                logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                return None

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = set()
            for i, node in enumerate(pending_draft_nodes):
                futures.add(executor.submit(execute_draft_node, node))
                logger.info(f"📤 Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
                    time.sleep(10)
                    logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

            initial_step_tasks = min(max_workers, total_steps - completed) - len(pending_draft_nodes)
            if initial_step_tasks > 0:
                for _ in range(initial_step_tasks):
                    futures.add(executor.submit(step_task))
                    logger.info(f"📤 Submitted initial step_task to fill thread pool")

            while completed < total_steps:
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                if not done:
                    continue  # timeout, no completed futures, retry (allows SIGINT handling)

                for fut in done:
                    futures.remove(fut)
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                        else:
                            logger.warning(f"⚠️  Task returned None (execution failed)")
                    except Exception as e:
                        logger.exception(f"❌ Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1  # Exclude virtual node
                        if completed == total_steps:
                            logger.info(journal_to_string_tree(journal))

                    if completed + len(futures) < total_steps:
                        futures.add(executor.submit(step_task, cur_node))
                        logger.info(f"📤 Submitted next task based on node {cur_node.id if cur_node else 'None'}")
                    logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            interpreter.terminate_all_subprocesses()
            executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
            raise
        finally:
            if not interrupted:
                executor.shutdown(wait=True)
    else:
        logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

    interpreter.cleanup_session(-1)


if __name__ == "__main__":    
    run()
