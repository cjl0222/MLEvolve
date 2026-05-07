"""Base code generation mode (single-shot plan + code).

The simplest generation strategy: one LLM call produces a natural language
plan followed by a complete code block. Used as the default / fallback mode
when diff or stepwise generation is not enabled or fails.
"""

from __future__ import annotations

import logging
from typing import Tuple

from llm import generate
from utils.response import extract_code, extract_text_up_to_code
import re

logger = logging.getLogger("MLEvolve")


# ============ Response format prompt (rewrite mode specific) ============

RESPONSE_FORMAT = {
    "Response format": (
        "Your response should be a brief outline/sketch of your proposed solution in natural language, "
        "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
        "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    )
}


# def plan_and_code_query(
#     agent_instance,
#     prompt,
#     retries: int = 3,
# ) -> Tuple[str, str]:
#     """Generate plan + code in one LLM call; returns (nl_text, code). On failure returns ("", raw_completion_text)."""
#     completion_text = None
#     for _ in range(retries):
#         completion_text = generate(
#             prompt=prompt,
#             temperature=agent_instance.acfg.code.temp,
#             cfg=agent_instance.cfg,
#         )
#         code = extract_code(completion_text)
#         nl_text = extract_text_up_to_code(completion_text)

#         if code and nl_text:
#             return nl_text, code

#         logger.debug("Extraction retry...")

#     logger.warning("Code extraction failed after retries")
#     return "", completion_text  # type: ignore

def plan_and_code_query(
    agent_instance,
    prompt,
    retries: int = 3,
) -> Tuple[str, str]:
    """Generate plan + code in one LLM call; returns (nl_text, code). On failure returns ("", raw_completion_text)."""
    
    completion_text = None
    last_syntax_errors = []
    
    for _ in range(retries):
        completion_text = generate(
            prompt=prompt,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
        )
        code, syntax_errors = extract_code(completion_text)
        nl_text = extract_text_up_to_code(completion_text)

        if code and nl_text and not syntax_errors:
            return nl_text, code

        if syntax_errors:
            last_syntax_errors = syntax_errors
            logger.debug(f"Syntax errors in generated code: {'; '.join(syntax_errors)}, retrying...")
        else:
            logger.debug("Extraction retry...")
            

    logger.warning("Code extraction failed after retries")
    
    
    # 有语法错误时：把错误信息存入 plan，把有问题的代码块存入 code
    # 这样 debug_agent 能看到具体是哪行语法错误，而不是一片空白
    if last_syntax_errors:
        error_summary = "; ".join(last_syntax_errors)
        # 重新提取第一个代码块（即使有语法错误）作为 code，供 debug_agent 参考
        raw_blocks = re.findall(r"```(?:python)?\n*(.*?)\n*```", completion_text, re.DOTALL)
        buggy_code = raw_blocks[0] if raw_blocks else completion_text
        return f"[CODE EXTRACTION FAILED] SyntaxError in generated code: {error_summary}", buggy_code
    
    return "", completion_text  # type: ignore