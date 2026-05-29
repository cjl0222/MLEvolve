"""Build guidance description for agent from skills.json via IO filter + RRF (BM25 + BGE) ranking at case level."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("MLEvolve")

INIT_SOLUTION_JSON = Path(__file__).resolve().parent / "init_solution_paths.json"

# ---------------------------------------------------------------------------
#  JSON helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
#  Tokenizer  (shared by BM25 and keyword matching)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase + split on non-alphanumeric (CJK-aware)."""
    return re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())


# ---------------------------------------------------------------------------
#  BM25
# ---------------------------------------------------------------------------


def _bm25_score(query: str, corpus_texts: List[str]) -> List[float]:
    """Simplified BM25 (TF * smooth-IDF)."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0] * len(corpus_texts)

    tokenized_corpus = [_tokenize(t) for t in corpus_texts]
    N = len(tokenized_corpus)
    if N == 0:
        return []

    df = {}
    for tokens in tokenized_corpus:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: max(0.0, (N - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}

    scores = []
    avg_dl = sum(len(t) for t in tokenized_corpus) / max(N, 1)
    k1, b = 1.5, 0.75
    for tokens in tokenized_corpus:
        doc_len = len(tokens)
        tf_local = {}
        for tok in tokens:
            tf_local[tok] = tf_local.get(tok, 0) + 1
        score = 0.0
        for qt in set(query_tokens):
            if qt not in idf:
                continue
            tf = tf_local.get(qt, 0)
            denom = tf + k1 * (1 - b + b * doc_len / avg_dl)
            score += idf[qt] * tf / denom
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
#  BGE embedding (lazy singleton)
# ---------------------------------------------------------------------------

_embedding_model = None  # EmbeddingModel instance
_embedding_model_path_used: Optional[str] = None


def _get_embedding_model(model_path: str) -> Any:
    """Lazy-load BGE embedding model (singleton). Returns None on failure."""
    global _embedding_model, _embedding_model_path_used
    if _embedding_model is not None and _embedding_model_path_used == model_path:
        return _embedding_model
    if not model_path or not Path(model_path).exists():
        logger.warning(f"[Coldstart] Embedding model path not found: {model_path}")
        return None
    try:
        from agents.memory.embedding_models import EmbeddingModel
        _embedding_model = EmbeddingModel(model_type="local", model_name=model_path)
        _embedding_model_path_used = model_path
        logger.info(f"[Coldstart] Loaded BGE embedding model from {model_path}")
        return _embedding_model
    except Exception as e:
        logger.warning(f"[Coldstart] Failed to load embedding model: {e}")
        return None


def _bge_similarity(
    query: str,
    corpus_texts: List[str],
    model: Any,
) -> List[float]:
    """Cosine similarity between query embedding and corpus embeddings."""
    if model is None or not corpus_texts:
        return [0.0] * len(corpus_texts)
    try:
        query_emb = model.encode([query])[0]
        corpus_embs = model.encode(corpus_texts)
        # Cosine similarity
        q_norm = np.linalg.norm(query_emb) + 1e-10
        c_norms = np.linalg.norm(corpus_embs, axis=1) + 1e-10
        similarities = np.dot(corpus_embs, query_emb) / (c_norms * q_norm)
        return similarities.tolist()
    except Exception as e:
        logger.warning(f"[Coldstart] BGE encoding failed, fallback to zeros: {e}")
        return [0.0] * len(corpus_texts)


# ---------------------------------------------------------------------------
#  RRF fusion
# ---------------------------------------------------------------------------


def _rrf_fusion(
    bm25_scores: List[float],
    bge_scores: List[float],
    k: int = 60,
) -> List[float]:
    """
    Reciprocal Rank Fusion on two ranked lists.

    RRF_score(i) = 1/(k + rank_bm25(i)) + 1/(k + rank_bge(i))

    Returns fused scores (higher = better), same length as inputs.
    """
    n = len(bm25_scores)
    if n == 0:
        return []

    # Rank 1 = best (highest score). Ties: average rank.
    def _ranks(scores: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        ranks = [0.0] * n
        j = 0
        while j < n:
            k_end = j
            while k_end + 1 < n and scores[order[k_end + 1]] == scores[order[j]]:
                k_end += 1
            avg_rank = (j + k_end) / 2.0 + 1.0  # 1-indexed
            for idx in range(j, k_end + 1):
                ranks[order[idx]] = avg_rank
            j = k_end + 1
        return ranks

    bm25_ranks = _ranks(bm25_scores)
    bge_ranks = _ranks(bge_scores)

    fused = []
    for i in range(n):
        score = 1.0 / (k + bm25_ranks[i]) + 1.0 / (k + bge_ranks[i])
        fused.append(score)
    return fused


# ---------------------------------------------------------------------------
#  SkillMatcher
# ---------------------------------------------------------------------------

# A matched case is a (skill_dict, case_dict) pair.
MatchedCase = Tuple[Dict, Dict]


class SkillMatcher:
    """
    Skills-based guidance builder — case-level matching.

    Pipeline:
      1. IO filter: exp_id → competition_tag → filter skills by skill.io and
         expand into (skill, case) pairs filtered by case.io.
      2. RRF (BM25 + BGE cosine) ranking on each (skill, case) pair.
      3. time_cost reorder hook (reserved, no-op currently).
      4. Build guidance text from top-k (skill, case) pairs.
    """

    def __init__(
        self,
        task_json_path: str,
        skills_json_path: str,
        embedding_model_path: Optional[str] = None,
        rrf_k: int = 60,
        top_k: int = 3,
    ):
        self.tasks: Dict = {}
        self.skills: List[Dict] = []
        self.rrf_k = rrf_k
        self.top_k = top_k
        self._embedding_model = None

        # Load task tags
        if task_json_path:
            try:
                self.tasks = _load_json(task_json_path)
            except Exception:
                logger.warning(f"[SkillMatcher] Could not load tasks from {task_json_path}")

        # Load skills
        if skills_json_path:
            try:
                data = _load_json(skills_json_path)
                self.skills = data.get("skills", [])
            except Exception:
                logger.warning(f"[SkillMatcher] Could not load skills from {skills_json_path}")

        # Lazy-load embedding model
        if embedding_model_path:
            self._embedding_model = _get_embedding_model(embedding_model_path)

    # ------------------------------------------------------------------
    #  IO filter → (skill, case) pairs
    # ------------------------------------------------------------------

    def _io_filter(self, exp_id: str) -> List[MatchedCase]:
        """
        Filter (skill, case) pairs whose skill.io and case.io both contain
        the task's category tag.

        Returns list of (skill_dict, case_dict) tuples.
        """
        if not self.skills:
            return []

        # Determine io tag from exp_id
        io_tag = ""
        if self.tasks and exp_id in self.tasks:
            io_tag = self.tasks[exp_id].strip()

        if not io_tag:
            # No tag → expand all skills into all (skill, case) pairs
            logger.info("[SkillMatcher] No IO tag found; using all (skill, case) pairs as candidates.")
            pairs: List[MatchedCase] = []
            for s in self.skills:
                for c in s.get("cases", []):
                    pairs.append((s, c))
            return pairs

        # Filter by io tag on both skill.io and case.io (case-insensitive)
        io_tag_norm = io_tag.casefold()
        pairs: List[MatchedCase] = []
        for s in self.skills:
            skill_io_tags = [t.casefold() for t in s.get("io", [])]
            if io_tag_norm not in skill_io_tags:
                continue  # skill doesn't apply to this IO category
            for c in s.get("cases", []):
                case_io_tags = [t.casefold() for t in (c.get("io", []) if isinstance(c.get("io"), list) else [c.get("io", "")])]
                # Also support single-string io
                if isinstance(c.get("io"), str):
                    case_io_tags = [c.get("io", "").casefold()]
                if io_tag_norm in case_io_tags:
                    pairs.append((s, c))

        if not pairs:
            logger.info(f"[SkillMatcher] IO tag '{io_tag}' matched no (skill, case) pair; falling back to all pairs.")
            for s in self.skills:
                for c in s.get("cases", []):
                    pairs.append((s, c))
        else:
            logger.info(f"[SkillMatcher] IO filter '{io_tag}' → {len(pairs)} (skill, case) pairs from {len(self.skills)} skills.")

        return pairs

    # ------------------------------------------------------------------
    #  Time-cost reorder  (reserved interface)
    # ------------------------------------------------------------------

    @staticmethod
    def _time_cost_reorder(ranked_pairs: List[MatchedCase]) -> List[MatchedCase]:
        """
        Reorder by time_cost (reserved interface).

        Currently a no-op passthrough. Future: promote low-cost skills.
        """
        return ranked_pairs

    # ------------------------------------------------------------------
    #  Build search texts for a (skill, case) pair
    # ------------------------------------------------------------------

    @staticmethod
    def _case_search_text(skill: Dict, case: Dict) -> str:
        """Build a retrieval-searchable text for a (skill, case) pair."""
        parts = [
            skill.get("name", ""),
            skill.get("description", ""),
            " ".join(skill.get("applicable_problems", [])),
            " ".join(skill.get("io", [])),
            case.get("description", ""),
            case.get("para", ""),
            case.get("applicable_problems", ""),
        ]
        return " ".join(parts)

    # ------------------------------------------------------------------
    #  Main match method
    # ------------------------------------------------------------------

    def match(
        self,
        exp_id: str,
        task_desc: Optional[str] = None,
    ) -> List[MatchedCase]:
        """
        Match (skill, case) pairs for the given task.

        Returns top_k (skill_dict, case_dict) tuples ordered by RRF score.
        """
        if not self.skills:
            return []

        # --- Phase 1: IO filter → (skill, case) candidate pool ---
        candidates = self._io_filter(exp_id)

        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates  # single candidate, no ranking needed

        # --- Phase 2: RRF ranking on each (skill, case) pair ---
        query = task_desc or exp_id or ""

        # Build search texts for each (skill, case) pair
        search_texts = [self._case_search_text(s, c) for s, c in candidates]

        # Rank A: BM25
        bm25_scores = _bm25_score(query, search_texts)

        # Rank B: BGE cosine similarity
        if self._embedding_model:
            bge_scores = _bge_similarity(query, search_texts, self._embedding_model)
        else:
            bge_scores = [0.0] * len(candidates)
            logger.info("[SkillMatcher] No BGE model; using BM25 only.")

        # RRF fusion
        rrf_scores = _rrf_fusion(bm25_scores, bge_scores, k=self.rrf_k)

        # Sort by RRF score
        ranked_indices = sorted(range(len(candidates)), key=lambda i: rrf_scores[i], reverse=True)
        top_items = [candidates[i] for i in ranked_indices[:self.top_k]]

        # --- Phase 3: time_cost reorder (no-op) ---
        top_items = self._time_cost_reorder(top_items)

        logger.info(
            f"[SkillMatcher] RRF top-{len(top_items)} pairs: "
            + ", ".join(
                f"{s.get('id', s.get('name', '?'))}/{c.get('io', '?')}"
                for s, c in top_items
            )
        )
        return top_items

    # ------------------------------------------------------------------
    #  Guidance text builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_guidance(matched_pairs: List[MatchedCase]) -> str:
        """
        Build the final guidance text consumed by draft_agent.py.

        Each (skill, case) pair produces one recommendation entry.
        Format compatible with the original 'Model1/Model2 ...' template.
        """
        if not matched_pairs:
            return "None model"

        lines = []
        for i, (skill, case) in enumerate(matched_pairs):
            skill_name = skill.get("name", f"Skill {i+1}")
            case_io = case.get("io", "")
            case_desc = case.get("description", "")
            case_problem = case.get("applicable_problems", "")

            # Build model label: "SkillName — case.applicable_problems"
            if case_problem:
                label = f"{skill_name} — {case_problem}"
            else:
                label = f"{skill_name} — {case_io}" if case_io else skill_name

            # Extract code_template from case
            code = case.get("code_template", "")

            lines.append(f"\nModel{i+1}: {label}\n")
            if case_desc:
                lines.append(f"Description:{case_desc}\n")
            if code:
                lines.append(
                    "Code template (MUST copy exactly — do NOT change model variant names or file paths):\n```python\n"
                    + code
                    + "\n```"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def build_guidance_description(
    cfg: Any,
    task_desc: Optional[str] = None,
) -> str:
    """
    Build coldstart guidance description for the agent.

    Single entry-point: IO filter → RRF (BM25 + BGE) at case level → top-k guidance.
    """
    task_json_path = getattr(cfg.coldstart, "task_json_path", "")
    skills_json_path = getattr(cfg.coldstart, "skills_json_path", "engine/coldstart/skills.json")
    rrf_k = int(getattr(cfg.coldstart, "rrf_k", 60))
    top_k = int(getattr(cfg.coldstart, "top_k", 3))

    torch_hub_dir = (getattr(cfg, "torch_hub_dir", "") or "").rstrip("/")
    exp_id = getattr(cfg, "exp_id", "") or ""

    # Resolve effective description (task_desc or cfg.goal)
    goal = getattr(cfg, "goal", None)
    effective_desc = task_desc or (str(goal) if goal else exp_id)

    # Embedding model path: reuse memory_embedding_model_path
    embedding_model_path = getattr(cfg.agent, "memory_embedding_model_path", "")

    matcher = SkillMatcher(
        task_json_path=task_json_path,
        skills_json_path=skills_json_path,
        embedding_model_path=embedding_model_path,
        rrf_k=rrf_k,
        top_k=top_k,
    )

    matched_pairs = matcher.match(exp_id, effective_desc)
    text = SkillMatcher.build_guidance(matched_pairs)

    # Replace torch hub dir placeholder if present
    if torch_hub_dir:
        text = text.replace("{TORCH_HUB_DIR}", torch_hub_dir)

    return text


def get_init_solution_paths(exp_id: str) -> List[str]:
    """Load init solution paths for exp_id from engine/coldstart/init_solution_paths.json."""
    if not INIT_SOLUTION_JSON.exists():
        return []
    try:
        data = _load_json(str(INIT_SOLUTION_JSON))
        paths = data.get(exp_id)
        if isinstance(paths, list):
            return [str(p) for p in paths if p]
        return []
    except Exception:
        return []