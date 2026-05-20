"""Stepwise code generation mode.

Provides stepwise code generation using multi-agent collaboration where specialized
agents handle different stages of the ML pipeline:
  - data_processing_and_feature_engineering
  - model_design
  - training_evaluation

Main entry: stepwise_plan_and_code_query()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from llm import generate, compile_prompt_to_md
from utils.response import extract_code, extract_text_up_to_code, wrap_code
from agents.planner.base_planner import (
    PLANNING_ALLOWED_MODULES,
    PLANNING_JSON_FORMAT,
    PLANNING_JSON_SCHEMA,
    parse_planning_response,
)
import re

logger = logging.getLogger("MLEvolve")


@dataclass
class StepwiseContext:
    stage: str = "draft"
    memory: str = ""
    previous_code: str = ""
    execution_output: str = ""


@dataclass
class StepAgent:
    name: str
    introduction: str
    description: str
    guidelines: List[str]

    def generate(
        self,
        task_desc: str,
        data_preview: str,
        previous_steps: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 3,
        improvement_mode: bool = False,
        previous_module_code: str = "",
        improvement_strategy: str = "",
    ) -> Tuple[str, str]:
        prompt = self._build_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview,
            previous_steps=previous_steps,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
            improvement_mode=improvement_mode,
            previous_module_code=previous_module_code,
            improvement_strategy=improvement_strategy,
        )

        completion_text = None
        # add by
        last_syntax_errors = []
        
        for _ in range(retries):
            completion_text = generate(
                prompt=prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            code, syntax_errors  = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text and not syntax_errors:
                return nl_text, code
            
            
            if syntax_errors:
                last_syntax_errors = syntax_errors
                logger.debug(f"Syntax errors in {self.name}: {'; '.join(syntax_errors)}, retrying...")
            else:
                logger.debug(f"Extraction retry for {self.name}...")
        
        # 可能包含语法错误或生成错误
        logger.warning(f"Code extraction failed after retries for {self.name}")
        
        # merge 阶段会看到每个 step 的 plan 和 code，把语法错误摘要放进 plan 后，MetaAgent 有机会据此修正。
        # 这里不是 debug_agent 直接参与。但若 step 阶段失败信息被保留，最终合并代码失败后进入主流程 debug 时，上下文更完整，后续排错更容易。
        # 核心价值是：避免静默丢失错误信息，让 merge 和后续 debug 都有线索。
        if last_syntax_errors:
            error_summary = "; ".join(last_syntax_errors)
            raw_blocks = re.findall(r"```(?:python)?\n*(.*?)\n*```", completion_text, re.DOTALL)
            buggy_code = raw_blocks[0] if raw_blocks else completion_text
            return f"[CODE EXTRACTION FAILED] ({self.name}) SyntaxError: {error_summary}", buggy_code
        
        
        return "", completion_text  # type: ignore

    def _build_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        previous_steps: List[Dict[str, str]], # # 前面步骤(特征工程、模型设计、训练评估)的结果列表，每项含 name/plan/code
        prompt_base: Dict[str, Any], # draft_agent 构建的基础 prompt（含 Introduction/Instructions/Memory）
        agent_instance,  # AgentSearch 实例，用于读取配置
        context: StepwiseContext,  # 当前上下文：stage/memory/previous_code/execution_output
        improvement_mode: bool = False, # 是否是"针对某个模块"的精准改进模式
        previous_module_code: str = "", # improvement_mode=True 时，该步骤上一版本的代码
        improvement_strategy: str = "", # # improvement_mode=True 时，具体的改进策略描述
    ) -> str:
        
        # 第一部分：构建 Introduction
        base_intro = prompt_base.get("Introduction", "") # 从 prompt_base 取出 draft_agent 写的通用角色介绍（Kaggle Grandmaster 那段）

        if context.stage == "improve": # 判断当前是改进阶段还是草稿阶段
            if improvement_mode and previous_module_code: # 精准改进模式：有明确的"上一版本模块代码"和"改进策略"
                step_specific_intro = (
                    f"You are currently working on improving the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the improved code for this specific step, based on the previous module code and the improvement strategy provided below. "
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}"
                ) # self.name例如特征工程、模型设计、训练评估等模块，previous_module_code是之前这个模块的代码，improvement_strategy是根据执行结果总结的改进策略，可以是具体的改进点或者一些启发式的改进方向
            else:  # 整体改进模式：没有精准的模块代码，基于整体上下文改进
                step_specific_intro = (
                    f"You are currently working on the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the code for this specific step that aligns with the overall improvement strategy. "
                    f"Base your implementation on the previous solution and execution results provided below, ensuring it integrates well with the improved approach."
                )
        else: # draft 阶段：简单告知当前步骤，不要写其他步骤的代码
            step_specific_intro = (
                f"You are currently focusing on the '{self.name}' step of the solution. "
                f"Your task is to write ONLY the code for this specific step, not the complete solution."
            )
        # 最终 introduction = 通用角色介绍 + 步骤专属说明
        introduction = base_intro + "\n\n" + step_specific_intro
        
        # 第二部分：构建前置步骤摘要
        prev_summary = ""
        # 如果有前面步骤的结果（Step 2 能看到 Step 1，Step 3 能看到 Step 1+2）
        if previous_steps:
            prev_parts = []
            # 遍历每个前置步骤
            for step in previous_steps:
                # 格式化为 markdown：步骤名 + 计划 + 代码块
                prev_parts.append(f"### {step['name']}\n**Plan:** {step['plan']}\n**Code:**\n{wrap_code(step['code'])}")
            prev_summary = "\n\n".join(prev_parts)
        else:
            prev_summary = "This is the first step, no previous steps."
        
        # 处理预训练模型 guidelines
        guidelines_to_use = self.guidelines.copy()
        # 判断是否启用了 coldstart（预训练模型推荐）
        use_pretrain = (
            hasattr(agent_instance, 'use_coldstart') and
            agent_instance.use_coldstart and
            hasattr(agent_instance, 'coldstart_description') and
            agent_instance.coldstart_description != "None model"
        )
        # 在 draft 阶段注入预训练提示（improve 阶段已有上下文，不需要）
        if use_pretrain and context.stage == "draft":
            # model_design 步骤：强制优先使用预训练模型
            if self.name == "model_design":
                pretrain_emphasis = [
                    "**CRITICAL: You MUST prioritize using the recommended pretrained models provided in the Implementation guideline section below.**",
                    "The pretrained models are STRONGLY RECOMMENDED and should be your default first choice.",
                    "Only use custom architectures if the pretrained models are clearly unsuitable for this specific task."
                ]
                # 插到 guidelines 最前面，确保 LLM 优先看到
                guidelines_to_use = pretrain_emphasis + guidelines_to_use
            elif self.name == "data_processing_and_feature_engineering":
                # 数据处理步骤：提醒后续可能用预训练模型，数据格式要兼容 （这里要加定制模型的信息）
                pretrain_awareness = [
                    "**IMPORTANT: Be aware that pretrained models may be used in later steps. Consider the input requirements of common pretrained models (e.g., image size, normalization, data format) when preparing the data and engineering features.**",
                    "For image tasks, ensure data is prepared in a format compatible with standard pretrained models (e.g., PIL Image, numpy arrays, proper image sizes).",
                    "For text tasks, ensure text data is properly tokenized and formatted for potential transformer models.",
                ]
                guidelines_to_use = pretrain_awareness + guidelines_to_use

        guidelines_text = "\n".join([f"- {g}" for g in guidelines_to_use])
        
        # 第四部分：修改 Instructions，明确要求只写当前步骤的代码，并且如果有预训练模型推荐，要强调优先使用预训练模型
        prompt_instructions = prompt_base["Instructions"].copy()
        
        # 覆盖 Response format，强制 LLM 只输出当前步骤的代码
        prompt_instructions["Response format"] = (
            "Your response should be:\n"
            "1. A brief plan (2-3 sentences) describing what you will do in this step\n"
            "2. A single markdown code block (wrapped in ```) containing ONLY the code for this step\n"
            "IMPORTANT: Do NOT write code for other steps. Only write code for the current step."
        )
        
        # 以步骤名为 key，注入该步骤专属的 guidelines  例如 key = "model_design guidelines"
        prompt_instructions[f"{self.name} guidelines"] = [guidelines_text] # 可能是data_processing_and_feature_engineering/model_design/training_evaluation guideline
        
        # 如果 Instructions 里有 "Implementation guideline"（来自 draft_agent 的通用实现指南）
        if "Implementation guideline" in prompt_instructions:  
            base_impl_guideline = prompt_instructions["Implementation guideline"] # 来自draft_agent的run()里get_impl_guideline_from_agent
            # 追加步骤协作相关的额外要求，确保每个步骤的代码既要满足通用实现指南，又要满足与其他步骤协作的要求
            step_specific_impl = [
                "The code for this step must be self-contained and can be integrated with other steps.",
                "Use clear variable names that are consistent with previous steps.",
                "Do not duplicate code from previous steps - assume those parts already exist.",
                "Make sure to handle edge cases appropriately.",
            ]
            if isinstance(base_impl_guideline, list): 
                prompt_instructions["Implementation guideline"] = base_impl_guideline + step_specific_impl
            else:
                prompt_instructions["Implementation guideline"] = [base_impl_guideline] + step_specific_impl
        
        # 第五部分：组装 prompt dict
        prompt: Dict[str, Any] = {
            "Introduction": introduction, # 角色介绍 + 步骤说明
            "Task description": task_desc, # 任务描述（含 Evaluation 部分）
            "Data preview": data_preview_str, # 数据集文件预览
            # Memory 优先从 prompt_base 取，其次从 context 取，最后为空字符串
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""), # 来自 agent.virtual_root.fetch_child_memory(),
            "Previous steps": prev_summary, #   # 前置步骤的 plan + code
            "Current step": {
                "Name": self.name,                # 当前步骤名，如 "model_design"
                "Description": self.description, # 当前步骤描述
            },
            "Instructions": prompt_instructions,  # 修改后的指令集
        }
        
        # improve 阶段需要额外注入"上一版本代码"
        if context.stage == "improve":
            # improvement_mode=True 和 previous_module_code 没有外部调用方传入
            if improvement_mode and previous_module_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(previous_module_code),
                    "Note": f"This is the previous code for the '{self.name}' module. Improve it based on the improvement strategy provided above."
                }
            elif "Previous solution" in prompt_base: # stepwise_plan_and_code_query() 目前只在 draft_agent.py:143-152 被调用，传入的是 draft prompt（没有 Previous solution），且 context.stage="draft"。
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code: # 同一次调用里 context 只传了 stage 和 memory，见 draft_agent.py:148-151。
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }
        
        # 第六部分：构建 assistant_suffix
        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)
        
        # 根据 stage 决定 okay_text 和 assistant_suffix
        if context.stage == "draft":
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            assistant_suffix = ""
        elif context.stage == "improve": # stepwise_coder 里的 elif context.stage == "improve" 基本不会走到，属于预留分支。
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            if improvement_mode and previous_module_code:
                previous_module_code_wrapped = wrap_code(previous_module_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously implemented the '{self.name}' module with the following code:\n{previous_module_code_wrapped}\n"
                    f"The execution of the full solution yielded the following results:\n{execution_output_wrapped}\n"
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}\n"
                    f"I need to improve this specific module according to the strategy above, ensuring it integrates well with the other modules."
                )
            elif context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            assistant_suffix = ""

        model_name = agent_instance.acfg.code.model.lower()
        
        # 第七部分：最终拼接
        # 构建 memory 段落（只有 Memory 非空时才加入）
        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"
        # 构建上一版本代码段落（只在 improve 阶段且有 Previous solution 时加入），上面代码中赋值
        previous_solution_section = ""
        if context.stage == "improve" and "Previous solution" in prompt:
            previous_solution_section = f"\n# Previous solution\n{prompt['Previous solution']['Code']}\n"
        
        # 拼接 user_prompt（用户消息部分）
        user_prompt = (
            f"\n# Task description\n{prompt['Task description']}\n\n"
            f"{memory_section}\n"
            f"{previous_solution_section}"
            f"# Previous steps\n{prompt['Previous steps']}\n\n"
            f"# Current step: {prompt['Current step']['Name']}\n{prompt['Current step']['Description']}\n\n"
            f"{instructions}"
        )
        return f"{introduction}\n\n{user_prompt}\n\n{okay_text}\n{prompt['Data preview']}{assistant_suffix}"



@dataclass
class MetaAgent:
    def merge(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 3,
    ) -> Tuple[str, str]:
        prompt = self._build_merge_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview_str,
            step_results=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
        )

        completion_text = None
        last_syntax_errors = []

        for _ in range(retries):
            completion_text = generate(
                prompt=prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            code, syntax_errors = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text and not syntax_errors:
                return nl_text, code

            if syntax_errors:
                last_syntax_errors = syntax_errors
                logger.debug(f"Syntax errors in MetaAgent merge: {'; '.join(syntax_errors)}, retrying...")
            else:
                logger.debug("Extraction retry for MetaAgent merge...")

        logger.warning("Code extraction failed after retries for MetaAgent merge")
        
        # add by 增加了语法错误的识别
        if last_syntax_errors:
            error_summary = "; ".join(last_syntax_errors)
            raw_blocks = re.findall(r"```(?:python)?\n*(.*?)\n*```", completion_text, re.DOTALL)
            buggy_code = raw_blocks[0] if raw_blocks else completion_text
            return f"[CODE EXTRACTION FAILED] (merge) SyntaxError: {error_summary}", buggy_code

        return "", completion_text

    def _build_merge_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]], # 各子步骤输出（name/plan/code）
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext, # 上下文（draft/improve、memory）
        ) -> Tuple[str, str]:
        introduction = (
            "You are a domain expert proficient in surrogate model modeling, an expert in writing clean, efficient, and high-performance Python code for ML tasks. "
            "You have received code snippets from a team of specialized agents, each focusing on a specific part of the ML pipeline. "
            "Your critical task is to intelligently merge these partial scripts into a single, cohesive, and fully runnable Python script."
        )

        steps_summary = []
        for i, result in enumerate(step_results, 1):
            steps_summary.append(f"""
        ### Step {i}: {result['name']}
        **Plan:** {result['plan']}
        **Code:**
        {wrap_code(result['code'])}
        """)

        prompt_instructions = prompt_base["Instructions"].copy()

        prompt_instructions["Response format"] = (
            "Your response should be a brief summary (2-3 sentences) of how you merged the steps, "
            "followed by a single markdown code block (wrapped in ```) containing the complete merged code. "
            "There should be no additional headings or text in your response."
        )

        prompt_instructions["Merge guidelines"] = [
            "- Combine all code sections into a single, runnable Python script",
            "- CRITICAL: You are a MERGER, not a designer. Faithfully integrate the code from all steps. Do NOT introduce new models, algorithms, or approaches that were not in the original steps.",
            "- Ensure variable names are consistent across steps",
            "- Remove duplicate imports and definitions",
            "- Resolve conflicts between steps by following the earlier step's design (e.g., model_design defines the model, training_evaluation trains it)",
            "- Ensure the execution flow is logical: data processing & feature engineering -> model design -> training & evaluation",
            "- Make sure the final code prints validation metric (must match task's Evaluation section) and saves submission.csv",
            "- The code should be a single-file Python program that can be executed as-is",
            "- Assume previous steps have NOT been executed; do not skip execution steps and only read files or outputs.",
            "- All parts must work together seamlessly",
        ]

        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Data preview": data_preview_str,
            "Step results": "".join(steps_summary),
            "Instructions": prompt_instructions,
        }

        if context.stage == "improve": # 预留分支 目前draft_agent中的stepwise_plan_and_code_query传入的是 "stage": "draft"  虽然 improve 阶段确实会构造 context["stage"] = "improve"（improve_agent.py:290），但那条路径走的是 diff/full-rewrite，不会进 stepwise 合成入口（improve_agent.py:239-247）。
            if "Previous solution" in prompt_base:
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

        okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"

        if context.stage == "improve":  # 也是预留分支，目前不会走到
            if context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            memory_section = f"# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}"
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            assistant_suffix = ""

        user_prompt = (
            f"\n# Task description\n{prompt['Task description']}\n\n"
            f"{memory_section}\n\n"
            f"# Step results\n{prompt['Step results']}\n\n"
            f"{instructions}"
        )
        return f"{introduction}\n\n{user_prompt}\n\n{okay_text}\n{prompt['Data preview']}{assistant_suffix}"


    def _simple_concat(self, step_results: List[Dict[str, str]]) -> str:
        code_parts = []
        for result in step_results:
            code_parts.append(f"# Step: {result['name']}\n{result['code']}\n")
        return "\n".join(code_parts)


def create_default_step_agents() -> List[StepAgent]:
    return [
        StepAgent(
            name="data_processing_and_feature_engineering",
            introduction="You are a Data Preparation Specialist responsible for data loading, cleaning, and feature engineering.",
            description="Load data from `./input` directory, perform cleaning, feature engineering, and create train/validation/test splits.",
            guidelines=[
                "Your responsibility: Load data from `./input`, clean, create features (preprocessing, encoding, augmentation), and split dataset into train/validation/test.",
                "CRITICAL: This step MUST include BOTH data loading AND feature engineering. Do NOT only load the raw data. You must actively create, transform, and enhance features to improve model performance.",
                "IMPORTANT: Apply feature engineering techniques such as feature scaling, encoding, transformation, and data augmentation methods appropriate for the task. Explore and implement feature engineering strategies that can enhance the model's ability to learn from the data.",
                "CRITICAL: Do NOT build models, write training code, or perform evaluation. Focus ONLY on data preparation and feature engineering.",
            ],
        ),
        StepAgent(
            name="model_design",
            introduction="You are a Model Architect responsible for designing the model architecture, loss function, and optimizer.",
            description="Design the model architecture (including pretrained models), and define the loss function and optimizer.",
            guidelines=[
                "Your responsibility: Design the model architecture or choose reference pretrained model, loss function, and optimizer based on the task and the features from previous steps.",
                "CRITICAL: Do NOT write the training loop, data processing, or feature engineering code. Only define the model, criterion, and optimizer objects.",
                "IMPORTANT: Consider the task's evaluation metric (from the task description's Evaluation section) when designing the model. The model output format should be compatible with the required evaluation metric calculation.",
                "IMPORTANT: When designing custom model architectures, include appropriate regularization components (e.g., Dropout layers) to prevent overfitting.",
            ],
        ),
        StepAgent(
            name="training_evaluation",
            introduction="You are a Training and Evaluation Expert responsible for implementing training, validation, and submission generation.",
            description="Implement the training loop, validation, metric tracking, model saving, and generate submission file.",
            guidelines=[
                "Your responsibility: Write the training loop that uses the data, features, model, loss function, and optimizer from previous steps. Include validation, metric tracking, save the best model. Then load the best model, calculate validation metric (must match task's Evaluation section), perform test inference, and save `submission.csv` to `./submission/` directory.",
                "CRITICAL: Assume that all previous code steps have already been executed. You should start directly from the training step. Do NOT redefine or reload the data, features, model, loss function, or optimizer. These components are already defined and available from the previous steps.",
                "CRITICAL: You MUST use the variables and objects defined in previous steps AS-IS. Do NOT replace, redesign, or substitute them with different approaches. Your ONLY job is to write the training/evaluation code for what was already defined — not to introduce new models or pipelines.",
                "IMPORTANT: Your code should assume the data preprocessing, feature engineering, and model design steps have been completed. Simply use the existing variables without copying them.",
                "CRITICAL: Validation metric computation must use the same prediction method as test inference, using training data only as reference, to avoid data leakage and ensure the metric reflects true generalization performance.",
                "CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
                "CRITICAL: You MUST actively prevent overfitting. Do NOT only focus on validation set metrics, as this can easily cause the model to overfit. You can consider to use standard anti-overfitting techniques as default modeling strategies, including:",
                "  - Data augmentation (when applicable to the task)",
                "  - Early stopping (monitor validation metric and stop when it stops improving)",
                "  - Regularization (weight decay, L1/L2 regularization)",
                "  - Dropout (if using neural networks)",
                "  - Other appropriate regularization techniques for the specific model type",
                "CRITICAL: You MUST implement the exact evaluation metric as specified in the task description's 'Evaluation' section. Read the Evaluation section carefully and implement it precisely according to the exact formula, calculation steps, and aggregation method described.",
                "CRITICAL: You MUST NOT use dummy, simplified, or approximate metrics. The validation metric must be a REAL and COMPLETE implementation of the task's evaluation metric as described in the Evaluation section, not an approximation, placeholder, or simplified version.",
                "CRITICAL: If the Evaluation section specifies multiple thresholds, components, or aggregation steps, you MUST implement ALL of them. Do not skip any required calculation steps or use shortcuts.",
                "CRITICAL: The metric calculation must match the Evaluation section exactly - use the same matching criteria, the same formula, the same thresholds (if any), and the same aggregation method as specified.",
                "CRITICAL: The final line must be: `print(f'Final Validation Score: {{score}}')`. This is required for the score parser.",
            ],
        ),
    ]


def stepwise_plan_and_code_query(
    agent_instance,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
    ) -> Tuple[str, str]:
    logger.info("Using stepwise generation route.")

    stepwise_context = StepwiseContext(
        stage=context.get("stage", "draft"),
        memory=context.get("memory", ""),
        previous_code=context.get("previous_code", ""),
        execution_output=context.get("execution_output", ""),
    ) # 封装 stage/memory/previous_code/execution_output等上下文信息，方便在不同步骤的agent之间传递

    step_agents = create_default_step_agents()
    meta_agent = MetaAgent()

    step_results: List[Dict[str, str]] = []
    for idx, agent in enumerate(step_agents, 1):
        logger.info(f"Step {idx}/{len(step_agents)}: {agent.name}")

        plan, code = agent.generate(
            task_desc=prompt_base["Task description"],
            data_preview=data_preview,
            previous_steps=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=stepwise_context,
        ) # 当出现语法错误时，plan中包含的是预编译语法报错信息，code中包含的是原始生成的代码（可能有语法错误）。这样做的目的是保留错误信息，让后续的 merge 和 debug 都有线索，而不是静默丢失错误信息。

        step_results.append({
            "name": agent.name,
            "plan": plan,
            "code": code,
        })

    logger.info("Merging all steps...")
    final_plan, final_code = meta_agent.merge(
        task_desc=prompt_base["Task description"],
        data_preview_str=data_preview,
        step_results=step_results,
        prompt_base=prompt_base,
        agent_instance=agent_instance,
        context=stepwise_context,
    )

    logger.info("Stepwise generation completed.")

    return final_plan, final_code
