import logging
from . import openai_compatible as _openai
from .openai_compatible import FunctionSpec, OutputType, PromptType, compile_prompt_to_md, generate
from config import Config
logger = logging.getLogger("MLEvolve")


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    cfg:Config=None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        if len(system_message) > 1000:
            logger.info(f"system: {system_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        if len(user_message) > 1000:
            logger.info(f"user: {user_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    output, req_time, in_tok_count, out_tok_count, info = _openai.query(
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        cfg=cfg,
        **model_kwargs,
    )
    logger.info("---Query complete---", extra={"verbose": True})

    return output
