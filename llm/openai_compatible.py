"""OpenAI API compatible backend: function calling (query), streaming generation (generate),
   prompt compilation, retry logic, and function-calling specs."""

import json
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import backoff
import jsonschema
from dataclasses_json import DataClassJsonMixin
from funcy import notnone, once, select_values
from openai import OpenAI
from config import Config

logger = logging.getLogger("MLEvolve")

# ---------------------------------------------------------------------------
#  Type aliases
# ---------------------------------------------------------------------------
PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

# ---------------------------------------------------------------------------
#  Prompt & message helpers
# ---------------------------------------------------------------------------

@backoff.on_predicate(
    wait_gen=backoff.constant,
    interval=5,
    max_time=300,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    """Call *create_fn* with automatic retry on transient errors."""
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.warning(f"Retryable error: {e}\n{traceback.format_exc()}")
        return False


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

# ---------------------------------------------------------------------------
#  OpenAI client
# ---------------------------------------------------------------------------

_client: OpenAI = None  # type: ignore


@once
def _setup_openai_client(cfg: Config):
    global _client
    _client = OpenAI(
        api_key=cfg.agent.code.api_key,
        base_url=cfg.agent.code.base_url,
        timeout=1200.0
    )


def _convert_func_spec_to_openai_tool(func_spec: FunctionSpec):
    """Convert FunctionSpec to OpenAI Tool format."""
    return func_spec.as_openai_tool_dict


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client(cfg)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    # Build messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": compile_prompt_to_md(system_message)})
    if user_message:
        messages.append({"role": "user", "content": compile_prompt_to_md(user_message)})
    else:
        # Ensure at least one user message is present to avoid "No user query found" errors
        # This is required by some OpenAI-compatible APIs
        messages.append({"role": "user", "content": ""})

    # Build tools
    tools = None
    tool_choice = None
    if func_spec is not None:
        tools = [_convert_func_spec_to_openai_tool(func_spec)]
        tool_choice = func_spec.openai_tool_choice_dict

    t0 = time.time()
    logger.info(f"Querying OpenAI-compatible API with model: {filtered_kwargs.get('model')}")

    try:
        response = _client.chat.completions.create(
            model=filtered_kwargs.get("model"),
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=filtered_kwargs.get("temperature", 1.0),
            max_tokens=filtered_kwargs.get("max_tokens", 16384),
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }, 
        )
        req_time = time.time() - t0

        # Parse response
        if func_spec is None:
            output = response.choices[0].message.content or ""
            logger.info(f"OpenAI response: {output}", extra={"verbose": True})
        else:
            message = response.choices[0].message
            if message.tool_calls:
                output = {
                    "name": message.tool_calls[0].function.name,
                    "arguments": json.loads(message.tool_calls[0].function.arguments)
                }
                logger.info(f"OpenAI function call: {output}", extra={"verbose": True})
                output = output["arguments"]
            else:
                # Fallback to content if no tool calls
                output = message.content or {}
                logger.info(f"OpenAI response (no function call): {output}", extra={"verbose": True})

        in_tokens = response.usage.prompt_tokens if response.usage else 0
        out_tokens = response.usage.completion_tokens if response.usage else 0

        info = {
            "model": filtered_kwargs.get("model"),
            "created": int(time.time()),
        }

        return output, req_time, in_tokens, out_tokens, info

    except Exception as e:
        logger.error(f"Error calling OpenAI-compatible API: {e}")
        raise e


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
) -> str:
    """Streaming text generation via OpenAI-compatible API.

    Args:
        prompt: The text prompt to complete.
        cfg: Config instance (provides model name and initializes client).
        temperature: Sampling temperature (default 1.0).
        max_tokens: Max output tokens (default 16384).
        stop_tokens: Optional stop sequences.
        json_schema: Optional JSON schema for structured output.
        max_retries: Max retry attempts on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        The generated text.
    """
    _setup_openai_client(cfg)

    # Convert dict/list prompts to markdown string
    if prompt is not None and not isinstance(prompt, str):
        prompt = compile_prompt_to_md(prompt)

    logger.info(f"generate prompt: {prompt}", extra={"verbose": True})

    try:
        response = _client.chat.completions.create(
            model=cfg.agent.code.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else 1.0,
            max_tokens=max_tokens if max_tokens is not None else 16384,
            stop=stop_tokens,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }, 
        )

        full_text = response.choices[0].message.content or ""

        logger.info(f"generate response: {full_text}", extra={"verbose": True})
        return full_text

    except Exception as e:
        logger.warning(f"generate failed, retrying: {e}")
        if max_retries <= 1:
            logger.error("generate retry limit reached")
            raise
        time.sleep(retry_delay)
        return generate(prompt, cfg, temperature, max_tokens, stop_tokens, json_schema, max_retries - 1, retry_delay)
