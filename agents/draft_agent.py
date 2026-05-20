"""Draft Agent: initial plan and code draft."""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.coder import plan_and_code_query, stepwise_plan_and_code_query
from agents.triggers import register_node
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_prompt_environment,
    get_impl_guideline_from_agent,
)
from agents.planner import build_chat_prompt_for_model

logger = logging.getLogger("MLEvolve")


def run(agent, init_solution_path: Optional[str] = None) -> SearchNode:
    """Generate initial draft. If init_solution_path is provided and readable, use file content directly."""
    if init_solution_path: # 如果传入了 init_solution_path，就先尝试直接读取这个文件内容
        try:
            code = Path(init_solution_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read init_solution from {init_solution_path}: {e}, falling back to LLM generation")
            init_solution_path = None
        else:
            plan = "User-provided init solution."
            agent.virtual_root.add_expected_child_count()
            new_node = SearchNode(
                plan=plan,
                code=code,
                parent=agent.virtual_root,
                stage="draft",
                local_best_node=agent.virtual_root,
            )
            register_node(agent, new_node, "User-provided init solution (no LLM).", new_branch=True)
            logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id}) [init_solution]")
            return new_node

    professional_identity = ( 
        "You are a domain expert proficient in surrogate model modeling—a top-tier machine learning specialist skilled in leveraging machine learning and physical laws for physical field prediction tasks, with the goal of optimizing evaluation metrics for these tasks.\n\n"
        "Your standards:\n"
        "✓ Design a complete machine learning pipeline (Data → Model → Training → Inference) or (Data → Pre-trained Model → Fine-tuning → Inference).\n"
        "✓ Implement models that truly learn from data (rather than baseline scripts that merely use constants).\n"
        "✓ Generate predictions by performing real model inference on each sample.\n"
        "✓ Pursue state-of-the-art performance for surrogate models, rather than settling for simple baseline solutions.\n\n"
        "Your solution will be evaluated on real-world data; please approach this with professional rigor.\n\n"
    )

    background = (
        "**Physical Field Types** you need to handle: Fluid dynamics (velocity/pressure fields), Structural mechanics (stress/strain fields), Heat transfer (temperature fields), Electromagnetic fields, and Multi-physics coupling.\n"
        "Physical Laws/Governing Equations:\n"
        "**Scenario Classification and Input/Output Specifications**:\n\n"
        "1. **Parameters → Parameters**\n"
        "   - Description: Input a fixed-dimensional parameter vector; output performance metrics or target parameters in scalar or vector form.\n"
        "   - Input x: `[N, D_in]` (Number of samples × Input features) parameter configuration vector.\n"
        "   - Output y: `[N, D_out]` (Number of samples × Output features) performance scalars or optimization parameters.\n"
        "   - Example: AutoML hyperparameter optimization, Geometric dimensions → Lift coefficient.\n\n"
        "2. **Parameters → Structured Physical Fields**\n"
        "   - Description: Input a parameter vector (e.g., initial conditions, boundary conditions, material parameters); output a physical field defined on a structured grid (rectangular grid), potentially spanning multiple time steps.\n"
        "   - Input x: `[N, D_feat]` (Number of samples × Feature dimensions) initial/control parameter vector.\n"
        "   - Output y: `[N, H, W]` or `[N, H, W, T]` or `[N, H, W, T, C]` (Samples × Height × Width × (Time steps) × (Channels)).\n"
        "   - Example: Inflow parameters → Pressure field around a 2D airfoil.\n\n"
        "3. **Parameters → Unstructured Physical Fields on Fixed Geometry**\n"
        "   - Description: Input a parameter vector; output physical quantities defined on a mesh with a fixed number of nodes (unstructured grid, constant node count), potentially spanning multiple time steps.\n"
        "   - Input x: `[N, D_feat]` (Number of samples × Feature dimensions) operating condition/load parameter vector.\n"
        "   - Output y: `[N, P, T, C]` or `[N, P, C]` (Samples × Nodes × (Time steps) × Channels).\n"
        "   - Example: Load parameters → Stress distribution on a fixed grid.\n\n"
        "4. **Parameters → Unstructured Physical Fields on Variable Geometry**\n"
        "   - Description: Input a parameter vector plus geometric information (point cloud coordinates, SDF, normals, etc.); output physical quantities on the corresponding point cloud. The number of points varies per sample (due to changing geometry).\n"
        "   - Input x: `[N, D_feat]` (Operating condition parameters) + Geometric information for each sample `geometry: [P_i, 7]` (Number of points × 7: xyz + SDF + normals).\n"
        "   - Output y: `[P_i, C]` (Number of points × Channels).\n"
        "   - Example: Design parameters + Geometric point cloud → Surface pressure distribution on an airfoil.\n\n"
        "5. **Structured Physical Field → Structured Physical Field**\n"
        "   - Description: Input one or more structured physical fields (e.g., initial flow field, boundary conditions, parameter fields); output another structured physical field (e.g., evolved flow field, pressure distribution).\n"
        "   - Input x: `[N, H, W, C_in]` (Samples × Height × Width × Input channels) initial/boundary structured fields (may include masks/SDF/dimensionless numbers).\n"
        "   - Output y: `[N, H, W, T, C_out]` or `[N, H, W, C_out]` (Samples × Height × Width × (Time steps) × Output channels).\n"
        "   - Example: Initial temperature field → Evolved temperature field via heat conduction; Low-resolution flow field → High-resolution flow field (Super-resolution).\n\n"
        "Based on the specific task, you will select the appropriate scenario definition and strictly adhere to the corresponding input/output specifications.\n"
    )

    introduction = (
        professional_identity +
        background + 
        "The physical field surrogate model modeling task now begins officially. "
        "You are required to propose an excellent and innovative solution, "
        "and implement this solution in Python with master-level expertise in machine learning and expert-level understanding in physics. "
        "We will now provide the task description."
    )
    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Memory": agent.virtual_root.fetch_child_memory(),
        "Instructions": {},
    }
    prompt["Instructions"] |= prompt_resp_fmt()

    prompt["Instructions"] |= {
        "🔬 Critical: Scientific Approach to Design": [
            "",
            "Before designing your solution, you must answer three fundamental questions:",
            "",
            "1. **WHAT makes this task unique?**",
            "   - Not generic observations like 'it's a classification task'",
            "   - What SPECIFIC patterns, challenges, or domain characteristics?",

            "",
            "2. **WHY is your approach suitable for this task?**",
            "   - Not just 'this model is good' - explain the MATCH between approach and task",
            "   - What properties of your method address the task characteristics?",

            "",
            "3. **HOW will you validate your hypothesis?**",
            "   - What outcome would confirm your approach is right?",
            "   - What outcome would suggest you need to reconsider?",

            "",
            "---",
            "",
            "⚠️ This is not a template to fill - this is how scientists think.",
            "Blindly applying standard methods without understanding WHY is not acceptable.",
            "",
            "Your plan should naturally reflect this reasoning process.",
        ],
    }

    prompt["Instructions"] |= {
        "Solution sketch guideline": [
            "- This first solution design should be relatively simple — avoid complex ensemble strategies or extensive hyperparameter searches at this stage.\n",
            "- 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:\n",
            "  • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.\n",
            "  • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.\n",
            "  • **Step 2 - Choose Strategy**:\n",
            "    → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.\n",
            "    → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.\n",
            "  • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).\n",
            "  • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.\n",
            "- Don't propose the same modelling solution but keep the evaluation the same.\n",
            "- Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.\n",
            "- Propose an evaluation metric that is reasonable for this task.\n",
            "- Don't suggest to do EDA.\n",
            "- The data is already prepared in `./input` directory. No need to unzip files.\n",
        ],
        "Coding & Execution Guidelines (CRITICAL)": [
            "- **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.",
            "- **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.",
            "- **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser."
        ]
    }
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()

    if agent.use_coldstart and (agent.coldstart_description != "None model"):
        coldstart_guideline = [
            f"""
            **Pretrained Model Strategy**:

            • **Option A [RECOMMENDED]**: {agent.coldstart_description}
              → SOTA models with proven performance. Use for end-to-end fine-tuning OR as frozen feature extractors.

            • **Option B**: Alternative pretrained models if better suited to task characteristics.

            • **Option C**: Train from scratch / non-DL methods (only when pretraining provides no advantage).

            **CRITICAL: When using any recommended pretrained model (Option A), you MUST copy the Code template EXACTLY as provided — including model variant names, file paths, and checkpoint filenames. Only the listed weights are available locally; other variants will fail to load.**

            **Key Techniques**:
            1. **Feature Extractor Pattern**: If dataset is small or domain mismatch exists → Freeze backbone + train only final layers (or feed to XGBoost/SVM).

            2. **Mixed Precision (MANDATORY for pretrained models)**: Use `torch.cuda.amp` (autocast + GradScaler) to save memory. DO NOT manually convert to .half().

            3. **Avoid Timeouts**: #1 cause is slow data loading, NOT GPU model.
               • Use DataLoader with num_workers>=2, pin_memory=True (NOT raw for loops)
               • For large datasets + heavy backbones: Extract & cache features to disk (.npy/.h5)
            """
        ]
    else:
        coldstart_guideline = [""]

    prompt["Instructions"]["Implementation guideline"].extend(coldstart_guideline)
    prompt["Instructions"] |= get_prompt_environment()
    prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY

    instructions = f"\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

    user_prompt = f"\n# Task description\n{prompt['Task description']}{memory_section}\n{instructions}"
    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll examine the dataset:\n{agent.data_preview}"
    prompt_complete = build_chat_prompt_for_model(
        agent.acfg.code.model, introduction, user_prompt, assistant_prefix
    )
    agent.virtual_root.add_expected_child_count()

    if agent.use_stepwise_generation: # 4次1次LLM调用（3个步骤 + 1次合并）	分工协作，质量更高
        plan, code = stepwise_plan_and_code_query(
            agent_instance=agent,
            prompt_base=prompt,
            data_preview=agent.data_preview,
            context={
                "stage": "draft",
                "memory": prompt.get("Memory", ""),
            },
        )
    else: # 1次LLM调用	单次生成，简单快速
        plan, code = plan_and_code_query(agent, prompt_complete)
    new_node = SearchNode(plan=plan, code=code, parent=agent.virtual_root, stage="draft",
                        local_best_node=agent.virtual_root)
    register_node(agent, new_node, prompt_complete, new_branch=True)

    logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id})")
    return new_node
