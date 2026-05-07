"""Node selection: UCT select, get_exploration_weight, get_top_k_nodes_global, select_from_top_k_weighted, select_with_soft_switch."""

import logging
import random
import time
from typing import List

from engine.search_node import SearchNode
from engine.conditions import should_trigger_branch_fusion
logger = logging.getLogger("MLEvolve")


def _piecewise_decay(t, initial_C=1.414, T1=100, T2=200, alpha=0.01, lower_bound=0.7):
    """Piecewise decay: initial_C until T1, linear to lower_bound by T2, then lower_bound."""
    if t < T1:
        return initial_C
    elif T1 <= t <= T2:
        return max(initial_C - alpha * (t - T1), lower_bound)
    else:
        return lower_bound


def _compute_exploration_constant(agent):
    """Compute exploration constant C from search progress (piecewise decay)."""
    # 控制 UCT 公式里的探索系数 C，随搜索步数分段衰减： 
    # 步数 < T1：C = 1.414（标准 √2，充分探索） 步数 T1→T2：C 线性从 1.414 降到 0.7  步数 > T2：C = 0.7（偏向利用）
    # UCT 公式是：Q + C × sqrt(ln(N) / n)
    # C 大 → 探索项权重高 → 倾向选访问少的节点（广度优先）  C 小 → 利用项权重高 → 倾向选历史奖励高的节点（深度优先）
    dcfg = agent.cfg.agent.decay
    n1 = agent.scfg.num_drafts * (agent.scfg.num_improves ** 2)
    n2 = round(agent.acfg.steps * dcfg.phase_ratios[0])
    t1 = min(n1, n2)
    t2 = round(agent.acfg.steps * dcfg.phase_ratios[1])
    return _piecewise_decay(
        t=agent.current_step,
        initial_C=dcfg.exploration_constant,
        T1=t1,
        T2=t2,
        alpha=dcfg.alpha,
        lower_bound=dcfg.lower_bound,  # 下限，默认 0.7
    )


def select(agent, node: SearchNode):
    """UCT selection: recurse from node, return node to expand (root lock for drafts).
    系统支持同时跑多个 worker（比如 3 个并发），每个 worker 独立调用 select 选节点，然后并发执行代码。
    根节点的子节点全是 draft 节点（初始草稿），每个 draft 代表一条独立的搜索分支。
    
    三个worker分别选择各自的draft吗？那在worker1的分支里，只能由worker1选择该分支的某一个节点，而不会让woker2或woker3来选择这分支里的节点吗
    
    “阶段性独占”，不是永久绑定
    在同一轮并发选择时，3 个 worker 通常会各自拿到不同的 draft/fusion_draft 分支（根层有 lock 过滤）
    当 worker1 选中某个分支后，这个分支在该轮会被锁住，worker2/3 这时不会进入这条分支。
    等 worker1 那次执行完成并回溯后，draft/fusion_draft 会解锁，解锁后，worker2/3 在后续轮次也可以进入这条分支。
    所以不是“这个分支永远只属于 worker1”，而是“同一时刻只允许一个 worker 进入该 draft 分支”。
    """
    def _best_child(n: SearchNode) -> SearchNode:
        C = _compute_exploration_constant(agent)
        if agent.is_root(n): # 根节点：过滤掉已被锁定（lock=True）的子节点，防止并发时多个 worker 同时展开同一个 draft 节点。选中后立即加锁。
            # 根节点层：lock 防止多 worker 进入同一分支
            filtered_children = [child for child in n.children if not child.lock] # 避免多个worker都选更高UTC值的节点 多个 worker 各自负责不同的分支，并行探索不重叠。当这个 draft 分支的某个子节点完成执行并反向传播时，draft 节点的 lock 才被释放。这时候这条分支已经有了执行结果，其他 worker 可以再次选择它（继续在这条分支上 improve）。
            selected_node = n
            if len(filtered_children) > 0:
                selected_node = max(filtered_children,
                                    key=lambda child: child.uct_value(exploration_constant=C))
            if selected_node.stage in ["draft", "fusion_draft"]: # 根节点的子节点不一定都是 draft。 也有可能selected_node是根节点stage="root"，此时要生成新 draft，不能锁
                selected_node.lock = True
            return selected_node
        else: # 非根节点：直接从所有子节点里选 UCT 值最高的，无锁逻辑。
            # 分支内部：expected_child_count 防止同一节点被过度展开
            # improve/debug 节点在分支内部，reached_child_limit 已经通过 expected_child_count 计数器控制了并发展开数量（worker 开始前 +1，失败时 -1），不需要额外的 lock 机制
            return max(n.children, key=lambda child: child.uct_value(exploration_constant=C))

    while node and not node.is_terminal:
        if not node.reached_child_limit(scfg=agent.scfg): # 子节点没满
            if node.is_buggy and node.is_debug_success is True: # buggy 且已调试成功
                node = _best_child(node) # 调试成功的节点已有好的子节点，继续深入
            elif node.continue_improve and len(node.children) > 0: # 有改进潜力且有子节点
                node = _best_child(node) # 这条分支还在改进中，跟着走
            else: # 其他（未满且无特殊标记）
                logger.info(f"[select] → node {node.id} (method=expand)")
                return node # 直接返回当前节点	这就是要展开的节点，让 agent 生成新子节点
        else: # 子节点满了
            if agent.is_root(node) and should_trigger_branch_fusion(agent) and random.random() < agent.acfg.branch_fusion_trigger_prob: # 根节点满了 + 聚合条件
                logger.info(f"Root node {node.id} is fully expanded for regular drafts, aggregation conditions met (including probability), returning root")
                return node # 触发分支聚合后，直接返回根节点，让 agent 从根节点开始生成新的 fusion_draft 分支（融合了多个已满的 regular draft 的优点）。
            node = _best_child(node) # 子节点满了（普通情况）	往下走	满了就不在这里展开，继续向下找叶节点
    logger.info(f"[select] → node {node.id} (method=uct)")
    return node


def get_exploration_weight(time_elapsed: float, total_time: float,
                           switch_start: float = 0.5,
                           switch_end: float = 0.7,
                           min_weight: float = 0.2) -> float:
    """Exploration weight: 1.0 until switch_start, linear decay to min_weight by switch_end."""
    time_progress = time_elapsed / total_time

    if time_progress < switch_start:
        return 1.0
    elif time_progress < switch_end:
        decay_progress = (time_progress - switch_start) / (switch_end - switch_start)
        return 1.0 - (1.0 - min_weight) * decay_progress
    else:
        return min_weight


def get_top_k_nodes_global(agent, k: int, max_from_same_branch: int) -> List[dict]:
    """Select top-k nodes globally with branch diversity (recomputed each call). Returns list of {node, branch_id, metric, rank}."""
    # 从所有分支中选出全局最优的 k 个节点，同时保证分支多样性
    # 第一步：收集所有有效节点（非 buggy，且 metric 有效）。这些节点可能分布在不同的分支上。
    all_nodes = []
    for branch_id in agent.branch_all_nodes: # branch_all_nodes 是一个 Dict[int, List[SearchNode]]，按分支 ID 组织所有节点。
        for node in agent.branch_all_nodes[branch_id]:
            if not node.is_buggy and node.metric is not None and node.metric.value is not None:
                all_nodes.append(node)

    if not all_nodes:
        logger.warning("No valid nodes found for Top-K selection")
        return []
    
    # 第二步：全局排序
    maximize = agent.metric_maximize
    all_nodes.sort(
        key=lambda n: n.metric.value,
        reverse=maximize
    ) # maximize=True（如 accuracy）→ 降序，最大值排前面  maximize=False（如 loss）→ 升序，最小值排前面

    logger.info(f"Total valid nodes: {len(all_nodes)}, requesting Top-{k}")

    selected = []
    branch_count = {}
    
    # 第三步：带多样性约束的贪心选取
    # 按 metric 从好到坏遍历，但每个分支最多贡献 max_from_same_branch 个节点。
    # 搜索树里不同分支代表不同的解题思路（不同的 draft 生成的代码方向）。如果 Top-K 全来自同一分支，后续的 improve 只是在一个思路上反复打磨，可能陷入局部最优。保留多个分支的代表节点，相当于同时押注多条赛道，增加找到全局最优解的概率。
    for node in all_nodes:
        if len(selected) >= k:
            break

        branch_id = node.branch_id
        current_count = branch_count.get(branch_id, 0)

        if current_count >= max_from_same_branch:
            logger.debug(f"Branch {branch_id} reached limit ({max_from_same_branch}), skipping node with metric={node.metric.value:.4f}")
            continue

        selected.append({
            'node': node,
            'branch_id': branch_id,
            'metric': node.metric.value,
            'rank': len(selected) + 1
        }) # rank 字段在 select_from_top_k_weighted 里直接用于计算采样概率：rank=1 的节点权重最高，被选中概率最大。
        branch_count[branch_id] = current_count + 1

    if selected:
        branch_distribution = {} # 用于打印
        for item in selected:
            bid = item['branch_id']
            branch_distribution[bid] = branch_distribution.get(bid, 0) + 1

        metrics_str = ", ".join([f"Rank{item['rank']}={item['metric']:.4f}(B{item['branch_id']})" for item in selected])
        logger.info(f"📊 Top-{len(selected)} selected: {metrics_str}")
        logger.info(f"📊 Branch distribution: {branch_distribution}")

    return selected


def select_from_top_k_weighted(agent, top_k_nodes: List[dict]) -> SearchNode:
    """Weighted random choice from top-k nodes (weight = 1/rank).
       从 Top-K 候选节点中按排名加权随机抽一个。
    """
    if not top_k_nodes:
        return select(agent, agent.virtual_root)

    weights = [1.0 / item['rank'] for item in top_k_nodes] # Top-K 为空（比如搜索早期还没有有效节点）时，退回标准 UCT
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    selected = random.choices(top_k_nodes, weights=probabilities)[0]

    logger.info(f"🎯 Selected: Rank{selected['rank']} (Branch {selected['branch_id']}, "
                f"metric={selected['metric']:.4f}, prob={probabilities[top_k_nodes.index(selected)]:.1%})")

    return selected['node']


def select_with_soft_switch(agent) -> SearchNode:
    """Soft switch: exploration (UCT) vs exploitation (Top-K) by time progress.
    这个函数是整个搜索策略的决策入口，解决的核心问题是：下一步该从哪个节点展开？
    它实现了强化学习中经典的 Exploration vs Exploitation 权衡，并随时间动态调整。
    """
    if agent.search_start_time is None:
        logger.info("📊 Search not started yet, using standard UCT")
        return select(agent, agent.virtual_root)

    time_elapsed = time.time() - agent.search_start_time
    total_time = agent.acfg.time_limit
    time_progress = time_elapsed / total_time

    scfg = agent.scfg

    exploration_weight = get_exploration_weight(
        time_elapsed, total_time,
        switch_start=scfg.explore_switch_start, # 默认 0.5
        switch_end=scfg.explore_switch_end,     # 默认 0.7
        min_weight=scfg.min_exploration_weight,  # 默认 0.2
    ) # 时间进度 0%  → 50%：weight = 1.0（纯探索）  时间进度 50% → 70%：weight 线性从 1.0 降到 0.2  时间进度 70% → 100%：weight = 0.2（主要利用） 

    if random.random() < exploration_weight: # 随机抽签决定本次走哪条路：探索（UCT）还是利用（Top-K）。探索模式的概率随时间衰减，前期更倾向探索，后期更倾向利用。
        logger.info(f"📊 Exploration mode (weight={exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")
        return select(agent, agent.virtual_root) # 探索模式：直接从根节点开始 UCT 选择，找到最值得展开的叶节点。这个过程不考虑全局 metric，只关注当前分支的 UCT 值，保持搜索的多样性和广度。

    else:
        # Top-K exploitation 利用模式：先全局选出 metric 最好的 k 个节点（跨分支去重，保证多样性），再从这 k 个节点里加权随机选一个（rank 越高概率越大）。如果这个节点未满，可以直接返回它；如果已满，则从这个节点开始 UCT 选择，找到它子树里最值得展开的叶节点返回。利用模式更关注全局 metric，快速聚焦在表现好的分支上，同时通过 Top-K 保持一定的多样性避免过早收敛。
        logger.info(f"🎯 Exploitation mode (weight={1-exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")

        if time_progress < scfg.explore_switch_end:
            k = scfg.topk_early_k
            max_from_same_branch = scfg.topk_early_max_per_branch
            phase = f"early-mid (<{scfg.explore_switch_end:.0%})"
        else: # 越到后期，Top-K 的 k 越大，允许从更多候选节点中选，同时限制同一分支最多贡献几个节点（保证多样性）。
            k = scfg.topk_late_k
            max_from_same_branch = scfg.topk_late_max_per_branch
            phase = f"late (>={scfg.explore_switch_end:.0%})"

        logger.info(f"📊 Phase: {phase}, requesting Top-{k} (max {max_from_same_branch} per branch)")
        
        # 从所有分支中按 metric 排序，选出全局最优的 k 个节点（跨分支去重）。
        top_k_nodes = get_top_k_nodes_global(
            agent,
            k=k,
            max_from_same_branch=max_from_same_branch
        )

        if not top_k_nodes:
            logger.warning("No valid Top-K nodes found, fallback to standard UCT")
            return select(agent, agent.virtual_root)
        
        # Top-K 节点不一定都能继续展开，reached_child_limit 检查它的子节点数是否已达上限。
        available_nodes = [
            item for item in top_k_nodes
            if not item['node'].reached_child_limit(agent.scfg, for_topk=True)
        ]

        if available_nodes: # 有未满节点
            # 按 1/rank 加权随机选一个（rank=1 的最优节点被选中概率最高），直接返回。_topk_triggered = True 标记这个节点是利用模式触发的，后续 _run_single_step 会用更宽松的停滞阈值
            selected_node = select_from_top_k_weighted(agent, available_nodes)
            logger.info(f"✅ Selected unexpanded Top-K node {selected_node.id} (from {len(available_nodes)}/{len(top_k_nodes)} available)")
            selected_node._topk_triggered = True
            return selected_node
        else: # 所有 Top-K 节点都满了  
            # 先用 Top-K 加权选出一个"好的起点"，再从这个起点往下做 UCT，找到它子树里最值得展开的叶节点。这是两级选择：全局定位好分支 + 局部 UCT 精确定位节点。
            logger.info(f"⚠️ All Top-{len(top_k_nodes)} nodes fully expanded, will apply UCT from selected node")
            selected_node = select_from_top_k_weighted(agent, top_k_nodes)
            logger.info(f"Selected fully expanded node {selected_node.id}, applying UCT from it")
            uct_node = select(agent, selected_node)
            uct_node._topk_triggered = True
            return uct_node
