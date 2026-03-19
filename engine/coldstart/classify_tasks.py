"""Classify tasks in competition_tag.json via OpenAI-compatible API function calling (structured output)."""

import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

API_KEY = ""       # fill in your API key
BASE_URL = ""      # fill in your base URL
MODEL = ""         # fill in your model name

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DESC_DIR = ""      # fill in the dataset directory path, e.g. "/path/to/datasets"
OUTPUT_JSON = PROJECT_ROOT / "engine" / "coldstart" / "competition_tag_classified.json"

CATEGORIES = ["General Image", "Detection", "Segmentation", "NLP", "Audio", "Others"]

SYSTEM_PROMPT = """You are a machine learning task classifier.
Given a Kaggle competition's task name and description, classify it by calling the classify function.

Rules:
- "General Image": image classification, image regression, or other general image tasks that are NOT detection or segmentation
- "Detection": object detection, bounding box prediction, localization, 3D object detection — any task predicting bounding boxes or object locations
- "Segmentation": image segmentation, pixel-level labeling, mask prediction tasks
- "NLP": text classification, NER, QA, text generation, sentiment analysis, code understanding, or any text/language-based task
- "Audio": audio classification, speech recognition, music tagging, sound event detection, or any audio-based task
- "Others": tabular, time series, video, molecular, signal processing, recommendation, or anything not fitting above

You MUST call the classify function with your answer."""

CLASSIFY_FUNC = {
    "type": "function",
    "function": {
        "name": "classify",
        "description": "Submit the classification result for a Kaggle competition task",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": CATEGORIES,
                    "description": "The task category",
                }
            },
            "required": ["category"],
        },
    },
    "strict": True,
}


def load_task_desc(task_name: str) -> str:
    """Load task description from description_short.md."""
    desc_file = Path(DESC_DIR) / task_name / "prepared" / "public" / "description.md"
    if not desc_file.exists():
        return ""
    text = desc_file.read_text(encoding="utf-8")
    return text[:3000]


def classify_task(client: OpenAI, task_name: str, desc: str) -> str:
    """Classify a single task via OpenAI-compatible API function calling."""
    contents = f"{SYSTEM_PROMPT}\n\nTask name: {task_name}\n\nDescription:\n{desc}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": contents}
        ],
        tools=[CLASSIFY_FUNC],
        tool_choice={"type": "function", "function": {"name": "classify"}},
        temperature=1.0,
        max_tokens=256,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }, 
    )

    if response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        category = tool_call.function.arguments.get("category", "")
        if category in CATEGORIES:
            return category
        print(f"  [WARN] invalid category for {task_name}: '{category}'")
        return "Others"

    text = response.choices[0].message.content or ""
    print(f"  [WARN] no function call for {task_name}, got: '{text[:100]}'")
    return None


RETRIES = 3


def _single_call(client: OpenAI, task_name: str, desc: str, max_retries: int = 2) -> tuple:
    """Single vote call; retries when classify_task returns None."""
    for attempt in range(max_retries):
        try:
            result = classify_task(client, task_name, desc)
            if result is not None:
                return task_name, result
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [ERROR] {task_name}: {e}")
    return task_name, None


def main():
    # fill in your task names here
    task_names = []

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"=== Processing {len(task_names)} tasks (x{RETRIES} vote) ===\n")

    descs = {name: load_task_desc(name) for name in task_names}

    votes = {name: [] for name in task_names}
    all_jobs = []
    for name in task_names:
        if not descs[name]:
            votes[name] = ["Others"]
            continue
        for _ in range(RETRIES):
            all_jobs.append((name, descs[name]))

    print(f"  Total API calls: {len(all_jobs)}, max_workers=20\n")

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(_single_call, client, name, desc)
            for name, desc in all_jobs
        ]
        done = 0
        for future in as_completed(futures):
            name, category = future.result()
            if category:
                votes[name].append(category)
            done += 1
            if done % 20 == 0:
                print(f"  progress: {done}/{len(all_jobs)}")

    results = {}
    for name in task_names:
        v = votes[name]
        if not v:
            results[name] = "Others"
        else:
            winner = Counter(v).most_common(1)[0][0]
            if len(set(v)) > 1:
                print(f"  [VOTE] {name}: {v} -> {winner}")
            results[name] = winner
        print(f"  {name} -> {results[name]}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== All results ===")
    for name in task_names:
        print(f"  {name}: {results[name]}")

    print(f"\n=== Category stats ===")
    for cat in CATEGORIES:
        count = sum(1 for v in results.values() if v == cat)
        if count:
            print(f"  {cat}: {count}")
    print(f"  Total: {len(results)}")
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
