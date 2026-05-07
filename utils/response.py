import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


# def extract_code(text):
#     # 两阶段提取：先找标准 markdown 代码块，找不到再把整个文本当代码处理。
#     parsed_codes = []
    
#     # 第一阶段： 标准 markdown 代码块
#     matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL) # 只认 python，也可以没有（空字符串）
#     for match in matches:
#         code_block = match[1]
#         parsed_codes.append(code_block)
    
#     # 第二阶段： fallback（整个文本就是代码） 场景：LLM 直接输出了代码，没有加 ``` 包裹。
#     if len(parsed_codes) == 0:
#         matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL) # 捕获组0：可选的开头 ``` 或 ```python，整体可有可无  (.*?)	捕获组2：主体内容，非贪婪
#         if matches:
#             code_block = matches[0][2] # 取 matches[0][2] 是捕获组2，即主体内容
#             parsed_codes.append(code_block)

#     valid_code_blocks = [
#         format_code(c) for c in parsed_codes if is_valid_python_script(c)
#     ] # 过滤掉语法不合法的代码块
#     return format_code("\n\n".join(valid_code_blocks))


def extract_code(text):
    # 两阶段提取：先找标准 markdown 代码块，找不到再把整个文本当代码处理。
    parsed_codes = []
    
    # 第一阶段： 标准 markdown 代码块
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL) # 只认 python，也可以没有（空字符串）
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)
    
    # 第二阶段： fallback（整个文本就是代码） 场景：LLM 直接输出了代码，没有加 ``` 包裹。
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL) # 捕获组0：可选的开头 ``` 或 ```python，整体可有可无  (.*?)	捕获组2：主体内容，非贪婪
        if matches:
            code_block = matches[0][2] # 取 matches[0][2] 是捕获组2，即主体内容
            parsed_codes.append(code_block)

    valid_code_blocks = []
    syntax_errors = []
    for c in parsed_codes:
        try:
            compile(c, "<string>", "exec")
            valid_code_blocks.append(format_code(c))
        except SyntaxError as e:
            line_text = e.text.strip() if e.text else "N/A"
            syntax_errors.append(f"line {e.lineno}: {e.msg} ({line_text})")
    
    return format_code("\n\n".join(valid_code_blocks)), syntax_errors


def extract_text_up_to_code(s):
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip() # 只是提取所有代码块前的文本（第一个 ``` 之前）


def extract_plan_from_diff_response(text: str) -> str:
    if not text:
        return ""

    stop_tokens = [
        "<<<<<<< SEARCH",
        "< SEARCH",
        ">>>>>>> REPLACE",
        "=======",
        "```",
    ]

    def cut_at_stop(s: str) -> str:
        indices = [s.find(token) for token in stop_tokens if s.find(token) != -1]
        if indices:
            return s[: min(indices)]
        return s

    if "Fixed Code Plan:" in text:
        candidate = text.split("Fixed Code Plan:", 1)[1]
        return cut_at_stop(candidate).strip()

    if "Plan:" in text:
        candidate = text.split("Plan:", 1)[1]
        return cut_at_stop(candidate).strip()

    return cut_at_stop(text).strip()


def extract_review(text):
    parsed_codes = []

    matches = re.findall(r"```(json)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(json)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    if len(parsed_codes) == 0 or not parsed_codes[0].strip():
        json_objects = extract_jsons(text)
        if len(json_objects) > 0:
            return json_objects[0]
        raise ValueError(f"No JSON found in text")

    try:
        review = json.loads(parsed_codes[0].strip())
        return review
    except json.JSONDecodeError:
        json_objects = extract_jsons(text)
        if len(json_objects) > 0:
            return json_objects[0]
        raise


def format_code(code) -> str:
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code
