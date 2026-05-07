"""Post-execution validation: validate_executed_node (csv existence, metric=0.0, register success)."""

import logging

from engine.search_node import SearchNode
from utils.metric import WorstMetricValue

logger = logging.getLogger("MLEvolve")

_ZERO_METRIC_ANALYSIS = (
    "Performance is 0.0 (complete failure). This indicates fundamental issues that need debugging:\n"
    "1. Model architecture may be incorrect or not learning\n"
    "2. Data preprocessing might be broken (wrong format, normalization issues)\n"
    "3. Loss function or evaluation metric calculation may be faulty\n"
    "4. Training loop might not be updating weights properly\n"
    "5. Input data might not be loaded correctly\n\n"
    "Please review the code carefully to identify the root cause."
)


def validate_executed_node(agent, node: SearchNode):
    """Check submission.csv exists, metric=0.0 anomaly; register successful node to branch.
    
    在 result_parse_agent 已经给出 node.is_buggy / node.metric 之后，再做一次工程侧校验：

    如果已经 buggy，直接不处理
    必须产出 submission/submission_node.id.csv
    对 maximize 任务，metric==0.0 视为可疑失败（强制打回 debug）
    通过检查后，把节点登记进 agent.branch_successful_nodes[branch_id]，供后续“停滞判断/融合策略”使用
    
    """
    if node.is_buggy:
        return

    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
    if not submission_path.exists(): # 如果没有产出 submission.csv，视为 buggy（不管 metric 是多少，都认为结果无效），并且直接返回不继续后续检查了。
        node.is_buggy = True
        node.metric = WorstMetricValue()
        logger.info(f"Node {node.id} did not produce a submission.csv")
        return

    if node.metric.maximize and node.metric.value == 0.0: # 当任务是“分数越大越好”时，得分恰好 0 往往是严重异常（例如预测全错、输出退化成占位值等），所以直接判失败并引导进入 debug。
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = _ZERO_METRIC_ANALYSIS
        logger.warning(
            f"Node {node.id} has metric=0.0 (maximize=True), marking as buggy for debugging."
        )
        return

    if hasattr(node, 'branch_id') and node.branch_id: # 通过校验后，登记为该分支成功节点.后续 is_branch_stagnant(...) 就是基于这个列表判断分支是否停滞。
        if node.branch_id not in agent.branch_successful_nodes:
            agent.branch_successful_nodes[node.branch_id] = []
        agent.branch_successful_nodes[node.branch_id].append(node)
