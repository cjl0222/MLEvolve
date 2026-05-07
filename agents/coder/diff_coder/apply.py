"""Diff patch application with retry logic.

Provides the shared apply-and-retry pipeline used by all diff-based
code generation modes (debug / improve / evolution / fusion).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, Tuple, Optional

from .patcher import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")


def apply_diff_with_retry(
    diff_response: str,
    original_code: str,
    max_retries: int = 3,
    regenerate_fn=None,
) -> Tuple[Optional[str], int, str]:
    '''
     diff-coder 的“补丁执行器 + 自动纠错重试器”。
     拿到 LLM 产出的 SEARCH/REPLACE 文本后，尽力打到代码上；如果打不上，就告诉 LLM为什么失败并重试。
     
     返回三元组：

    final_code | None
    成功/部分成功：修改后的代码
    完全失败：None
    
    total_applied: int
    成功应用的 patch 数量
    
    retry_note: str
    失败或部分成功时给上层/LLM的诊断说明
    完全成功时通常是空字符串

    '''
    current_code = original_code  # 当前正在被改的代码版本
    total_applied = 0 # 累计应用了多少 patch block
    retry_note = "" # 给下一轮重试的提示
    current_response = diff_response # 当前待应用的 LLM 输出

    for attempt in range(max_retries):
        try:
            logger.info(f"Applying diff patches... (attempt {attempt + 1}/{max_retries})")

            if current_response and (
                "<<<<<<< SEARCH" in current_response
                or "< SEARCH" in current_response
                or "<<<<<<<" in current_response
            ):
                if "<<<<<<< SEARCH" in current_response:
                    search_markers = current_response.count("<<<<<<< SEARCH")
                    replace_markers = current_response.count(">>>>>>> REPLACE")
                elif "< SEARCH" in current_response:
                    search_markers = current_response.count("< SEARCH")
                    replace_markers = current_response.count("> REPLACE")
                else:
                    search_markers = 1
                    replace_markers = 0
                has_incomplete_block = search_markers > replace_markers # 开始块比结束块多，通常是 LLM 输出被截断了。

                patcher = SearchReplacePatcher()
                updated_code, count = patcher.apply_patch(current_response, current_code, strict=False) # 真正应用补丁  用 SearchReplacePatcher 把当前响应打到 current_code ;count 是本轮成功应用数量 ;
                if count > 0 and updated_code and updated_code != current_code: # 若本轮有变化，就更新 current_code 并累加 total_applied  
                    current_code = updated_code
                    total_applied += count

                if total_applied > 0 and current_code != original_code and not has_incomplete_block:
                    logger.info(f"Successfully applied {total_applied} diff patch(es)")
                    return current_code, total_applied, ""   # 最佳成功路径
                else:
                    if has_incomplete_block and (count > 0 or total_applied > 0):
                        retry_note = (
                            "Your previous diff output appears truncated/incomplete "
                            "(missing closing '>>>>>>> REPLACE'). "
                            f"I have already applied {total_applied} patch(es). "
                            "Please continue and provide ONLY the remaining patches."
                        )
                    else:
                        retry_note = (
                            "Your previous diff did not apply cleanly to the current code. "
                            "Please generate minimal SEARCH/REPLACE blocks that match the CURRENT code exactly."
                        )

                    logger.warning(
                        f"Diff attempt {attempt + 1}/{max_retries}: "
                        f"count={count}, total_applied={total_applied}, "
                        f"code_changed={current_code != original_code}, "
                        f"search_markers={search_markers}, replace_markers={replace_markers}, "
                        f"has_incomplete_block={has_incomplete_block}"
                    )

                    if attempt < max_retries - 1 and regenerate_fn:
                        logger.info("Regenerating diff...")
                        current_response = regenerate_fn(current_code, retry_note)
                        continue
                    else:
                        if total_applied > 0:
                            return current_code, total_applied, retry_note
                        return None, 0, retry_note
            else:
                retry_note = (
                    "Your previous output did not contain valid SEARCH/REPLACE blocks. "
                    "Output ONLY complete SEARCH/REPLACE blocks (no other text)."
                )
                logger.warning(
                    f"Diff attempt {attempt + 1}/{max_retries}: "
                    "Response does not contain SEARCH/REPLACE format"
                )

                if attempt < max_retries - 1 and regenerate_fn:
                    logger.info("Regenerating diff...")
                    current_response = regenerate_fn(current_code, retry_note)
                    continue
                else:
                    return None, 0, retry_note

        except Exception as e:
            logger.warning(f"Diff attempt {attempt + 1}/{max_retries} failed with exception: {e}")
            retry_note = (
                f"Your previous diff failed to apply due to an error: {e}. "
                "Please output minimal SEARCH/REPLACE blocks that match the CURRENT code exactly, "
                "and ensure every block is complete."
            )
            if attempt < max_retries - 1 and regenerate_fn:
                logger.info("Regenerating diff...")
                try:
                    current_response = regenerate_fn(current_code, retry_note)
                except Exception as retry_e:
                    logger.error(f"Failed to regenerate diff: {retry_e}")
                continue
            else:
                if total_applied > 0:
                    return current_code, total_applied, retry_note
                return None, 0, retry_note

    return None, 0, retry_note


def format_planning_result_for_plan(planning_result: Dict[str, Any]) -> str:
    """Serialize planning result for node plan storage."""
    try:
        return json.dumps(planning_result, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(planning_result)
