import logging
from typing import Any, List, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.prompts import prompt_resp_fmt, get_impl_guideline_from_agent
from agents.planner import build_chat_prompt_for_model
from agents.coder import plan_and_code_query

from engine.conditions import should_trigger_branch_fusion  # noqa: F401
from agents.triggers import register_node
from utils.metric import WorstMetricValue

logger = logging.getLogger("MLEvolve")


def _collect_branch_representatives(agent) -> List[SearchNode]:
    # 收集各分支代表节点
    representatives = []

    for branch_id, successful_nodes in agent.branch_successful_nodes.items(): # 只包含运行成功（无 bug）的节点，由 register_node 维护。
        if not successful_nodes or len(successful_nodes) == 0:
            logger.debug(f"Branch {branch_id} has no successful nodes, skipping")
            continue

        maximize = agent.metric_maximize if agent.metric_maximize is not None else True # None 时默认 True（越大越好）。
        # branch_best = max(
        #     successful_nodes,
        #     key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
        #         float("-inf") if maximize else float("inf")
        #     ),
        # )
        # 修复metric对象比较
        branch_best = max(
            successful_nodes,
            key=lambda n: n.metric if n.metric is not None and n.metric.value is not None else WorstMetricValue(),
        )

        if not branch_best.metric or branch_best.metric.value is None:
            logger.debug(f"Branch {branch_id} best node has no valid metric, skipping")
            continue

        representatives.append(branch_best)

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True
    # representatives.sort(
    #     key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
    #         float("-inf") if maximize else float("inf")
    #     ),
    #     reverse=maximize,
    # ) 把所有分支的代表节点按 metric 全局排序： 排序后，LLM 在 prompt 里会先看到最优分支，有助于它优先学习最成功的方案。
    representatives.sort(
        key=lambda n: n.metric if n.metric is not None and n.metric.value is not None else WorstMetricValue(),
        reverse=True,
    )

    logger.info(
        f"Collected {len(representatives)} branch representatives "
        f"from {len(agent.branch_successful_nodes)} successful solutions" # 记录"从 N 个分支中收集到 M 个代表"。注意日志里写的是 successful solutions 但实际是 branch_successful_nodes 的分支数，措辞略有歧义。
    )
    return representatives


def run(
    agent,
    mode: str = "node",
    parent_node: Optional[SearchNode] = None,
) -> Optional[SearchNode]:

    if parent_node and not agent.is_root(parent_node):
        # 当多条分支各自探索到瓶颈时，把它们的精华融合成一个全新的解决方案，开辟新的搜索分支。
        logger.error(
            f"_aggregation() should only be called from root node! Got parent_node: {parent_node.id}"
        )
        return None  # 只能从根节点触发

    if agent.fusion_draft_count >= agent.max_fusion_drafts:
        logger.info(
            f"Max fusion drafts ({agent.max_fusion_drafts}) reached, skipping aggregation"
        )
        return None  # 只能从根节点触发 限制 fusion 总次数，防止无限融合

    branch_representatives = _collect_branch_representatives(agent) # # 每个分支只取历史最优节点作为代表，然后按 metric 排序。至少需要 2 个分支才能做融合（单分支没有"融合"的意义）。
    if len(branch_representatives) < 2:
        logger.info("Not enough successful branches for aggregation")
        return None

    introduction = (
        "You are a domain expert proficient in surrogate model modeling. "
        "You are provided with multiple successful solutions from different independent branches below. "
        "Your task is to synthesize these diverse approaches and create a completely NEW solution "
        "that draws inspiration from their strengths. "
        "This is a fresh start to spark new ideas by combining insights from different successful directions."
    )

    reference_summaries = []
    if mode == "node": # # Node 模式（默认）：给 LLM 看每个分支最终最优节点的信息  LLM 看到的是"结果"，从成功方案中提炼共性   Node 模式强调"合并最终技术" 
        for i, node in enumerate(branch_representatives):
            trajectory = node.generate_node_trajectory(need_code=False)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Best Solution** (Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)
    elif mode == "trajectory": # Trajectory 模式：给 LLM 看每个分支从头到尾的演化路径（最多 6 步） LLM 看到的是"过程"，从演化路径中发现规律  Trajectory 模式强调"学习演化模式"
        for i, node in enumerate(branch_representatives):
            trajectory = node.get_root_to_current_trajectory(max_steps=6) # 展示的是从根到这个最优节点的路径，而不是整条分支的完整历史。
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Evolution Path** (Best Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)
    else:
        logger.warning(f"Unknown aggregation mode: {mode}, using node mode as default")
        for i, node in enumerate(branch_representatives):
            trajectory = node.generate_node_trajectory(need_code=False)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Best Solution** (Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)

    reference_experiences = "\n" + "-" * 80 + "\n".join(reference_summaries)

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Branch Experiences": reference_experiences,
        "Instructions": {},
    }

    prompt["Instructions"] |= prompt_resp_fmt()  # 响应格式

    if mode == "node":
        prompt["Instructions"] |= {
            "Multi-branch aggregation guideline (Node Mode)": [
                "- You are provided with the BEST solutions from different independent branches.",
                "- Analyze what makes each branch's final solution successful - their key techniques and approaches.",
                "- This is NOT about improving a current solution - this is about creating a FRESH NEW approach.",
                "- Think creatively: how can you synthesize the strengths of different final solutions into an innovative approach?",
                "- Write a brief natural language description of your NEW synthesized approach.",
                "- The solution should be distinct and innovative, combining the best ideas in a novel way.",
                "- Focus on discovering new synergies between successful techniques from different branches.",
                "- The final code should be a single, runnable Python script.",
                "- Do not suggest to do EDA.",
            ],
        }
    else:
        prompt["Instructions"] |= {
            "Multi-branch aggregation guideline (Trajectory Mode)": [
                "- You are provided with the EVOLUTION PATHS of different independent branches.",
                "- Analyze how each branch evolved from initial ideas to their best solutions - what worked and what didn't.",
                "- Learn from the successful improvement patterns and evolution strategies across branches.",
                "- This is NOT about improving a current solution - this is about creating a FRESH NEW approach.",
                "- Think creatively: what new directions emerge from understanding these different evolution paths?",
                "- Write a brief natural language description of your NEW synthesized approach.",
                "- The solution should be distinct and innovative, inspired by successful evolution patterns.",
                "- Focus on discovering unexplored directions suggested by the evolution insights from multiple branches.",
                "- The final code should be a single, runnable Python script.",
                "- Do not suggest to do EDA.",
            ],
        }
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    data_preview = getattr(agent, "data_preview", "") or ""
    assistant_prefix = (
        "Let me approach this systematically.\n"
        f"First, I'll examine the dataset:\n{data_preview}\n"
        "I have access to multiple successful approaches from different independent branches. "
        "I'll synthesize these diverse insights and create a completely new solution "
        "that combines the best ideas in an innovative way."
    )

    user_prompt = (
        f"\n# Task description\n{prompt['Task description']}\n\n"
        f"# Branch Experiences\n{prompt['Branch Experiences']}\n\n{instructions}"
    )
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix) # # 不是真正的 prefill

    plan, code = plan_and_code_query(agent, prompt_complete) # 修改了部分内容

    aggregation_node = SearchNode(
        plan=plan,
        code=code,
        parent=agent.virtual_root, # 挂在根节点下，是一条新分支的起点
        stage="fusion_draft",
        local_best_node=agent.virtual_root, # 初始化局部最优为根节点，表示这条新分支还没有自己的最优节点
    )
    register_node(agent, aggregation_node, prompt_complete, new_branch=True) # new_branch=True 让 register_node 为这个节点分配一个全新的 branch_id，它会开辟一条独立的搜索分支，后续的 improve/debug 都在这条新分支上进行
    agent.fusion_draft_count += 1

    logger.info(f"[aggregation] → node {aggregation_node.id} (branch={aggregation_node.branch_id})")
    return aggregation_node
