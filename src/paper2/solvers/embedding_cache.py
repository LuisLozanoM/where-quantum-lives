"""Embedding cache for rolling-window re-optimization.

Persists and reuses embeddings across consecutive windows when the
logical graph structure hasn't changed, avoiding redundant embedding
computation. Requires stable variable labels (QuboProblem.labels).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from paper1.embedding import embedding_reusable, find_embedding_for_graph, logical_graph_from_qubo
from paper1.formulations import QuboProblem
from paper1.qpu import sampler_working_graph


class EmbeddingCache:
    """Cache embeddings keyed by (N, graph_structure_hash, solver_name)."""

    def __init__(self, cache_dir: str | Path | None = None):
        self._cache: dict[str, dict] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _cache_key(self, problem: QuboProblem, solver_name: str) -> str:
        """Deterministic key based on N, nonzero structure, and solver.

        Uses SHA-256 instead of hash() for cross-session stability.
        """
        N = problem.size
        mask = np.abs(problem.matrix) > 1e-15
        np.fill_diagonal(mask, False)
        edge_count = int(np.count_nonzero(mask)) // 2
        digest = hashlib.sha256(
            f"{N}:{edge_count}:".encode() + mask.tobytes()
        ).hexdigest()[:16]
        return f"{solver_name}_n{N}_e{edge_count}_{digest}"

    def get(
        self,
        problem: QuboProblem,
        solver_name: str,
    ) -> dict | None:
        """Retrieve a cached embedding if one exists."""
        key = self._cache_key(problem, solver_name)
        return self._cache.get(key)

    def get_or_find(
        self,
        problem: QuboProblem,
        sampler,
        solver_name: str,
        *,
        seeds: tuple[int, ...] = (0, 1, 2),
        tries: int = 20,
    ) -> dict | None:
        """Return cached embedding or find a new one.

        Tries multiple seeds and keeps the embedding with the shortest
        mean chain length.
        """
        logical = logical_graph_from_qubo(problem)
        target = sampler_working_graph(sampler)

        cached = self.get(problem, solver_name)
        if cached is not None:
            if embedding_reusable(logical, target, cached):
                return cached

        best: dict | None = None
        best_mean = float("inf")

        for seed in seeds:
            try:
                emb = find_embedding_for_graph(
                    logical, target, random_seed=seed, tries=tries
                )
                if not emb:
                    continue
                mean_chain = float(np.mean([len(v) for v in emb.values()]))
                if mean_chain < best_mean:
                    best_mean = mean_chain
                    best = emb
            except Exception:
                continue

        if best is not None:
            self.put(problem, solver_name, best)

        return best

    def put(
        self,
        problem: QuboProblem,
        solver_name: str,
        embedding: dict,
    ) -> None:
        """Store an embedding in the cache."""
        key = self._cache_key(problem, solver_name)
        self._cache[key] = embedding
        if self._cache_dir:
            self._save_to_disk(key, embedding)

    def _save_to_disk(self, key: str, embedding: dict) -> None:
        """Serialize with JSON-safe string keys, preserving original type info."""
        path = self._cache_dir / f"{key}.json"
        # Store key type alongside value for lossless round-trip
        serializable = {}
        for k, v in embedding.items():
            serializable[json.dumps(k)] = v
        path.write_text(json.dumps(serializable))

    def _load_from_disk(self) -> None:
        if not self._cache_dir:
            return
        for path in self._cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                embedding = {json.loads(k): v for k, v in data.items()}
                self._cache[path.stem] = embedding
            except (json.JSONDecodeError, ValueError):
                continue

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
