"""Node evaluation: backpropagate, check_improvement, get_node_reward."""

import logging
import time
import random

from engine.search_node import SearchNode

logger = logging.getLogger("MLEvolve")


def backpropagate(node: SearchNode, value: float, add_to_tree=True):
    """Propagate reward up the tree; update debug_success, continue_improve, lock.
    而是 MCTS/树搜索里的奖励回传：
    从当前节点一路沿 parent 向上走到根，把这次结果影响传播给祖先节点。
    
     “阶段性独占”，不是永久绑定
    在同一轮并发选择时，3 个 worker 通常会各自拿到不同的 draft/fusion_draft 分支（根层有 lock 过滤）
    当 worker1 选中某个分支后，这个分支在该轮会被锁住，worker2/3 这时不会进入这条分支。
    等 worker1 那次执行完成并回溯后，draft/fusion_draft 会解锁，解锁后，worker2/3 在后续轮次也可以进入这条分支。
    所以不是“这个分支永远只属于 worker1”，而是“同一时刻只允许一个 worker 进入该 draft 分支”。
    
    为什么只解 draft / fusion_draft 的 lock？
    因为这个 lock 只在“根节点选择分支”时用于防并发撞车。
    加锁发生在 node_selection.py:51-60：

    只有根节点挑子节点时，会过滤 child.lock=True 的分支
    选中的 draft/fusion_draft 会被置 lock=True
    目的是让多个 worker 不要同时钻进同一条草稿分支
    所以在回溯时只需要对这两类节点解锁（evaluation.py:22-23），表示“这条分支这一轮执行完成了，允许别的 worker 再来选它”。
    
    
    其他 stage 的 lock 需要解吗？
    当前实现里不需要，因为其他 stage 基本没用这个 lock 机制。

    分支内部（improve/debug）的并发控制靠 expected_child_count / reached_child_limit，不是靠 lock。这在注释里也写了，见 node_selection.py:61-64。

    换句话说：

    根层分支并发：lock
    分支内部扩展并发：expected_child_count
    
    什么叫“回溯(backpropagate)”？
    当前节点拿到一个 reward 后，把这个 reward 逐级传给父、祖父、…、根节点，更新整条路径的搜索统计和状态。
    作用是让上层节点“知道”这条分支最近表现如何，从而影响后续 select 阶段的节点选择（探索/利用平衡）。
    一句话总结：
    get_node_reward 负责“打分”，backpropagate 负责“把分数沿树往上传”。这是树搜索里最关键的反馈闭环。
    """
    logger.info(f"[backprop] node {node.id}, reward={value}")
    while node is not None:
        if node.parent and node.is_buggy is False and node.parent.is_buggy is True:
            node.parent.is_debug_success = True
        elif node.parent and node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
            node.parent.is_debug_success = True  # 若子节点成功且父节点 buggy，则父 is_debug_success=True
        if node.parent and node.parent.stage != "root": 
            node.parent.continue_improve = node.continue_improve # 同步 continue_improve 给父节点（非 root 父） 标记该节点还有改进潜力，
        if node.stage in ["draft", "fusion_draft"] and node.lock:
            node.lock = False
        if node.improve_failure_depth > 0: # 连续改进失败计数器，如果新节点提升幅度低于阈值，就 +1 ；达到 max_improve_failure 后，把链条标为 terminal 并回溯
            node.improve_failure_depth = 0  # 一次链条收束后，失败深度重新计数，不把上一轮的“连败”带到下一轮路径里。
        node.update(value, add_to_tree) # 更新该节点统计量（如 visits / total_reward 等）
        node = node.parent


def get_node_reward(agent, node: SearchNode):
    reward = 0

    if node.is_buggy is True or node.is_buggy is None: # 失败或状态不完整 → reward = -1
        reward = -1
    elif node.is_buggy is False and node.metric.value is None:
        reward = -1
    else:
        if node.metric.value is not None and agent.best_metric is not None: # 如果比全局 best 更好：+1.5（强奖励）
            improvement = node.metric.value - agent.best_metric if node.metric.maximize else agent.best_metric - node.metric.value
            if improvement > 0:
                logger.info(f"Node {node.id} is better than the best node {agent.best_node.id} now!")
                reward += 1.5

        if node.parent and node.parent.stage != "root": # 再加“父节点关系奖励”：
            if node.parent.is_buggy is True: # 父节点 buggy（说明成功修复了坏链）→ +1.5
                reward += 1.5
            else: # 父节点非 buggy（普通改进） → +1
                reward += 1 
    return reward


def check_improvement(agent, cur_node: SearchNode, parent_node: SearchNode):
    '''
    
    整体分两段逻辑
        时间驱动的强制 backprop（中后期节奏控制）
        常规改进评估（基于 metric 提升幅度和失败深度）

    输入
    agent：全局状态与配置（时间限制、阈值、日志、journal 等）
    cur_node：刚执行并解析完结果的当前节点
    parent_node：它的父节点
    返回
    True：本轮已执行 backpropagate（该链条先收束）
    False：不回传，继续把 cur_node 放到 current_node_list 里做后续扩展
    '''
    
    
    improvement = 0
    should_backpropagate = False

    if (agent.search_start_time and
        cur_node.stage != "root" and
        cur_node.branch_id is not None): # 只有真正进入搜索后、非 root、且有分支 id 的节点才参与这一套。

        time_elapsed = time.time() - agent.search_start_time
        time_progress = time_elapsed / agent.acfg.time_limit # 计算时间进度

        if not hasattr(agent, 'branch_node_count'):
            agent.branch_node_count = {}

        branch_id = cur_node.branch_id
        agent.branch_node_count[branch_id] = agent.branch_node_count.get(branch_id, 0) + 1 # 该分支累计产生了第几个节点（current_count）
        current_count = agent.branch_node_count[branch_id]

        force_backprop = False

        scfg = agent.scfg
        # 两档强制策略  目的：避免一条链条无限深挖，提升收敛节奏和分支回流频率。
        if time_progress >= scfg.force_backprop_late_threshold: # 0.8 之后进入最后阶段，随机概率打回（节奏控制，增加多样性）
            if random.random() < scfg.force_backprop_late_prob: # 0.5 的概率强制打回  后期（>=80%）：按概率触发强制 backprop
                force_backprop = True
                logger.info(f"[Force Backprop] Late stage ({time_progress:.1%}), "
                        f"node {cur_node.id} (stage={cur_node.stage}, branch={branch_id}, #{current_count})")

        elif time_progress >= scfg.force_backprop_mid_threshold and current_count % scfg.force_backprop_mid_modulo == 0: # 中期（>=40%）：每逢该分支第 modulo 个节点触发（例如每 3 个一次）
            force_backprop = True
            logger.info(f"[Force Backprop] Mid stage ({time_progress:.1%}), "
                       f"branch {branch_id} node #{current_count}, "
                       f"node {cur_node.id} (stage={cur_node.stage})")

        if force_backprop: # 即使触发了 force_backprop，也可能被跳过：
            skip_force_backprop = False

            if (not cur_node.is_buggy and
                cur_node.metric is not None and
                cur_node.metric.value is not None): # 仅当当前节点非 buggy 且有有效 metric

                recent_window = scfg.recent_best_window
                recent_nodes = [n for n in agent.journal[-recent_window:]
                               if (not n.is_buggy and n.metric and n.metric.value is not None)] # 取最近窗口 recent_best_window（默认 4）中的有效节点

                if recent_nodes: # 若当前节点是“近期最佳”之一，则 skip_force_backprop=True
                    if cur_node.metric.maximize:
                        recent_best = max(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value >= recent_best.metric.value
                    else:
                        recent_best = min(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value <= recent_best.metric.value

                    if is_recent_best: # 刚出好结果时，允许继续延长改进链，不要被硬切断。
                        logger.info(f"[Smart Backprop] Node {cur_node.id} is recent best "
                                  f"(metric={cur_node.metric.value:.4f}), skip force backprop to continue improvement chain")
                        skip_force_backprop = True
            # 真正执行强制 backprop 前的 local_best 更新
            if not skip_force_backprop: # 若不 skip
                if (not cur_node.is_buggy and
                    cur_node.metric is not None and
                    cur_node.metric.value is not None):

                    local_best = cur_node.local_best_node  # 真正执行强制 backprop 前的 local_best 更新:若不 skip，会先尝试更新 cur_node.local_best_node（当前更优则替换）
                    if local_best and local_best.metric and local_best.metric.value is not None:
                        if agent.metric_maximize:
                            is_better = cur_node.metric.value > local_best.metric.value
                        else:
                            is_better = cur_node.metric.value < local_best.metric.value

                        if is_better:
                            cur_node.local_best_node = cur_node
                            logger.info(f"  └─ Updated local_best: {cur_node.metric.value:.4f} "
                                      f"(prev: {local_best.metric.value:.4f})")
                    else: # 如果之前 local_best 没有有效 metric，则当前节点只要有有效 metric 就直接设为 local_best
                        cur_node.local_best_node = cur_node
                        logger.info(f"  └─ Set as local_best: {cur_node.metric.value:.4f}")

                reward = get_node_reward(agent, cur_node)
                # 当前节点拿到一个 reward 后，把这个 reward 逐级传给父、祖父、…、根节点，更新整条路径的搜索统计和状态
                # 作用是让上层节点“知道”这条分支最近表现如何，从而影响后续 select 阶段的节点选择（探索/利用平衡）。
                backpropagate(cur_node, reward) 
                return True # 直接结束函数，表示本轮已回溯了，不继续后续改进评估了。
    # 常规改进评估（基于 metric 提升幅度和失败深度）
    local_best_node = cur_node.local_best_node
    local_best_metric = local_best_node.metric.value

    if cur_node.is_buggy is False: # 成功节点
        new_metric = cur_node.metric.value
        if parent_node.is_buggy: # 父节点是 buggy（debug 成功场景）
            logger.info(f"[eval] debug success for {parent_node.id}")
            if new_metric:
                if local_best_metric:
                    debug_improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
                    if debug_improvement > 0:
                        cur_node.local_best_node = cur_node # 与 local_best_metric 比较，若更好更新 local_best
                    cur_node.continue_improve = True
                    should_backpropagate = False # 倾向继续改进链条，暂不回溯
                else:
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
            else:
                should_backpropagate = True
        # 常规提升判断（核心阈值逻辑）
        if new_metric is not None and local_best_metric is not None: # 这部分是“微小改进容忍+连续失败终止”的主干策略。
            improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
            if improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < agent.scfg.max_improve_failure:
                local_best_node.improve_failure_depth += 1  # 提升不足 + 失败深度未达上限
                action = "continue"
                cur_node.continue_improve = True
            elif improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth >= agent.scfg.max_improve_failure:
                action = "terminal"   # 提升不足 + 失败深度已达上限
                cur_node.continue_improve = False
                should_backpropagate = True
                cur_node.is_terminal = True
            else: 
                action = "continue" # 提升足够
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement={improvement:.6f}, action={action}")
        elif new_metric is not None: # new_metric is not None 但 local_best_metric is None
            cur_node.local_best_node = cur_node # 直接把当前设为 local_best，继续
            cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=continue")
        else: # new_metric is None，说明虽然修复了 bug，但没有 metric 结果（可能是执行失败或 metric 计算失败），暂时先回溯。
            should_backpropagate = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
    elif cur_node.is_buggy is None: # 状态未决，保守处理：既不是成功也不是失败，可能是执行结果不完整或解析失败。为了安全起见，暂时先回溯，避免在不确定状态上继续扩展。
        logger.warning(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
        should_backpropagate = True
    else: # cur_node.is_buggy is True（失败节点）
        if cur_node.debug_depth >= agent.scfg.back_debug_depth: # debug_depth 超过 back_debug_depth（默认 2）后，认为这条链条已经深入挖掘过了，继续挖可能性价比不高了，先回溯。
            should_backpropagate = True
            if cur_node.debug_depth >= agent.scfg.max_debug_depth: # debug_depth 超过 max_debug_depth（默认 4）后，认为这条链条已经挖掘过头了，标为 terminal，后续不再考虑。
                cur_node.is_terminal = True

    if should_backpropagate:
        reward = get_node_reward(agent, cur_node)
        backpropagate(cur_node, reward)
    else:
        agent.current_node_list.append(cur_node) # 继续把当前节点放到 current_node_list 里做后续扩展（不回溯）current_node_list 是一个“可继续扩展节点”的缓存列表   语义上：它本来想表示“前沿候选节点池（continue 的节点） 实现上：目前基本是“只记不读”的状态，对主流程决策影响很小（甚至可能是遗留变量）
    return should_backpropagate
