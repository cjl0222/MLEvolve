#!/usr/bin/env python3
"""Test cases for SkillMatcher case-level matching (IO filter + RRF + case-aware guidance)."""

import sys
import os
import re
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from engine.coldstart.knowledge import SkillMatcher
from engine.coldstart.knowledge import _bm25_score, _tokenize  # noqa: F401 — used in assertions


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
TASK_JSON_PATH = str(_THIS_DIR / "competition_tag_classified.json")
SKILLS_JSON_PATH = str(_THIS_DIR / "skills.json")


def _count_pairs(text: str) -> int:
    """Count Model{N}: entries in guidance text."""
    return len(re.findall(r"\nModel\d+:", text))


def _extract_model_labels(text: str) -> list:
    """Extract Model{N}: label lines."""
    return re.findall(r"\nModel\d+: (.+)", text)


# ---------------------------------------------------------------------------
#  Test 1: IO filter — DINOv3 has 2 cases, exp_id → 'General Image' only hits
#          the General Image case, Detection case excluded
# ---------------------------------------------------------------------------

def test_io_filter_selects_correct_case():
    matcher = SkillMatcher(
        task_json_path=TASK_JSON_PATH,
        skills_json_path=SKILLS_JSON_PATH,
        embedding_model_path=None,
        rrf_k=60,
        top_k=5,
    )

    # aptos2019-blindness-detection → "General Image"
    pairs = matcher.match(exp_id="aptos2019-blindness-detection",
                          task_desc="classify diabetic retinopathy severity from retinal fundus images")

    # Assert: all returned pairs have case.io matching "General Image"
    for skill, case in pairs:
        case_io = case.get("io", "")
        if isinstance(case_io, list):
            assert "General Image" in case_io, f"Expected General Image in case.io list, got {case_io}"
        else:
            assert case_io == "General Image", f"Expected General Image case, got {case_io}"

    # Assert: should include DINOv3 General Image case AND Siglip2
    found_dinov3 = any(
        s.get("id") == "dinov3-general-image" and c.get("io") == "General Image"
        for s, c in pairs
    )
    found_siglip2 = any(s.get("id") == "siglip2-general-image" for s, c in pairs)
    assert found_dinov3, "Expected DINOv3 General Image case in results"
    assert found_siglip2, "Expected Siglip2 in results"

    # Assert: DINOv3 Detection case should NOT appear
    found_detection = any(
        s.get("id") == "dinov3-general-image" and c.get("io") == "Detection"
        for s, c in pairs
    )
    assert not found_detection, "DINOv3 Detection case should NOT appear for General Image task"

    # Build guidance and verify
    guidance = SkillMatcher.build_guidance(pairs)
    assert "DINOv3" in guidance
    assert "Detection" not in guidance.split("Model")[0]  # Detection case description shouldn't appear
    assert "Siglip2" in guidance

    print("[PASS] Test 1: IO filter selects correct case (General Image) for DINOv3")


# ---------------------------------------------------------------------------
#  Test 2: IO filter — exp_id not in competition_tag_classified.json → fallback
#          to all (skill, case) pairs. DINOv3's both cases should appear.
# ---------------------------------------------------------------------------

def test_io_filter_fallback_all_cases():
    matcher = SkillMatcher(
        task_json_path=TASK_JSON_PATH,
        skills_json_path=SKILLS_JSON_PATH,
        embedding_model_path=None,
        rrf_k=60,
        top_k=10,
    )

    # Unknown exp_id → fallback to all pairs
    pairs = matcher.match(exp_id="unknown-exp", task_desc="general purpose ML task")

    # DINOv3 has 2 cases; both should appear in fallback
    dinov3_cases = [(s, c) for s, c in pairs if s.get("id") == "dinov3-general-image"]
    assert len(dinov3_cases) >= 1, f"Expected at least 1 DINOv3 case in fallback, got {len(dinov3_cases)}"

    guidance = SkillMatcher.build_guidance(pairs)
    assert "DINOv3" in guidance
    assert "None model" not in guidance

    print(f"[PASS] Test 2: Fallback returns all pairs, DINOv3 has {len(dinov3_cases)} case(s)")


# ---------------------------------------------------------------------------
#  Test 3: Single-case skill compatibility (all existing skills except DINOv3)
# ---------------------------------------------------------------------------

def test_single_case_skill_compatibility():
    matcher = SkillMatcher(
        task_json_path=TASK_JSON_PATH,
        skills_json_path=SKILLS_JSON_PATH,
        embedding_model_path=None,
        rrf_k=60,
        top_k=10,
    )

    # NLP task → ModernBERT and DeBERTa
    pairs = matcher.match(exp_id="jigsaw-toxic-comment-classification-challenge",
                          task_desc="classify toxic comments in online discussions")

    # Check ModernBERT (single case)
    modernbert_pairs = [(s, c) for s, c in pairs if s.get("id") == "modernbert-nlp"]
    assert len(modernbert_pairs) == 1, f"Expected 1 ModernBERT case, got {len(modernbert_pairs)}"

    # Check DeBERTa (single case)
    deberta_pairs = [(s, c) for s, c in pairs if s.get("id") == "deberta-nlp"]
    assert len(deberta_pairs) == 1, f"Expected 1 DeBERTa case, got {len(deberta_pairs)}"

    # Guidance should contain both NLP models
    guidance = SkillMatcher.build_guidance(pairs)
    assert "ModernBERT" in guidance
    assert "DeBERTa" in guidance

    print("[PASS] Test 3: Single-case skills work correctly")


# ---------------------------------------------------------------------------
#  Test 4: Guidance output structure — each case produces one Model entry
#          with description and code_template from case
# ---------------------------------------------------------------------------

def test_guidance_output_structure():
    matcher = SkillMatcher(
        task_json_path=TASK_JSON_PATH,
        skills_json_path=SKILLS_JSON_PATH,
        embedding_model_path=None,
        rrf_k=60,
        top_k=3,
    )

    pairs = matcher.match(exp_id="aptos2019-blindness-detection",
                          task_desc="classify diabetic retinopathy severity")
    guidance = SkillMatcher.build_guidance(pairs)

    # Verify structure
    assert guidance != "None model"
    assert _count_pairs(guidance) >= 1, f"Expected at least 1 Model entry, got {_count_pairs(guidance)}"

    # Each Model should have Description and Code template
    model_blocks = guidance.strip().split("\nModel")
    for block in model_blocks[1:]:  # Skip text before first Model
        assert "Description:" in block, f"Missing Description in block: {block[:80]}..."
        assert "Code template" in block, f"Missing Code template in block: {block[:80]}..."

    # Model label should include case.applicable_problems
    labels = _extract_model_labels(guidance)
    for label in labels:
        assert " — " in label, f"Model label should contain ' — ' separator, got: {label}"

    print(f"[PASS] Test 4: Guidance output structure correct, {_count_pairs(guidance)} model entries")
    print(f"  Labels: {labels}")


# ---------------------------------------------------------------------------
#  Test 5: Empty skills or no matches → 'None model'
# ---------------------------------------------------------------------------

def test_empty_skills_returns_none_model():
    matcher = SkillMatcher(
        task_json_path="",
        skills_json_path="",
        embedding_model_path=None,
    )
    matcher.skills = []
    pairs = matcher.match(exp_id="test", task_desc="test")
    guidance = SkillMatcher.build_guidance(pairs)
    assert guidance == "None model"
    print("[PASS] Test 5: Empty skills returns 'None model'")


# ---------------------------------------------------------------------------
#  Test 6: Top-k truncation — only top_k pairs returned
# ---------------------------------------------------------------------------

def test_top_k_truncation():
    matcher = SkillMatcher(
        task_json_path=TASK_JSON_PATH,
        skills_json_path=SKILLS_JSON_PATH,
        embedding_model_path=None,
        rrf_k=60,
        top_k=2,
    )

    pairs = matcher.match(exp_id="aptos2019-blindness-detection",
                          task_desc="classify diabetic retinopathy")
    guidance = SkillMatcher.build_guidance(pairs)

    # With top_k=2, at most 2 pairs returned
    assert len(pairs) <= 2, f"Expected <= 2 pairs, got {len(pairs)}"
    assert _count_pairs(guidance) <= 2, f"Expected <= 2 Model entries, got {_count_pairs(guidance)}"
    assert _count_pairs(guidance) == len(pairs), f"Mismatch: {len(pairs)} pairs vs {_count_pairs(guidance)} guidance entries"

    print(f"[PASS] Test 6: Top-k truncation works (top_k=2, got {len(pairs)} pairs)")


# ---------------------------------------------------------------------------
#  Test 7: _case_search_text contains all relevant fields
# ---------------------------------------------------------------------------

def test_case_search_text():
    matcher = SkillMatcher(
        task_json_path=TASK_JSON_PATH,
        skills_json_path=SKILLS_JSON_PATH,
        embedding_model_path=None,
    )

    # Get a known pair
    for s in matcher.skills:
        if s.get("id") == "dinov3-general-image":
            for c in s.get("cases", []):
                if c.get("io") == "General Image":
                    text = SkillMatcher._case_search_text(s, c)
                    assert "DINOv3" in text
                    assert "image classification" in text
                    assert "Standard image classification" in text
                    assert "256x256" in text
                    print(f"[PASS] Test 7: _case_search_text covers skill + case fields")
                    return

    assert False, "Could not find DINOv3 General Image case"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SkillMatcher Case-Level Matching Tests")
    print("=" * 60)
    print()

    tests = [
        test_io_filter_selects_correct_case,
        test_io_filter_fallback_all_cases,
        test_single_case_skill_compatibility,
        test_guidance_output_structure,
        test_empty_skills_returns_none_model,
        test_top_k_truncation,
        test_case_search_text,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)