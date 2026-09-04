"""
Re-Ranker — Maximal Marginal Relevance (MMR)
=============================================

After SBERT retrieves top-N similar candidates, MMR re-ranks them to
balance RELEVANCE (similarity to query) with DIVERSITY (avoiding duplicates).

Usage:
    from knowledge.reranker import mmr_rerank

    results = mmr_rerank(query_emb, candidate_embs, candidate_texts, top_k=5)
"""

import torch
from sentence_transformers import util
from typing import List
from dataclasses import dataclass


@dataclass
class RankedResult:
    """A single re-ranked result with scoring details."""
    index: int              # Original index in the candidate list
    text: str               # The paragraph text
    relevance_score: float  # SBERT cosine similarity to query
    diversity_score: float  # How different this is from already-selected results
    final_score: float      # MMR combined score


def mmr_rerank(
    query_emb: torch.Tensor,
    candidate_embs: torch.Tensor,
    candidate_texts: List[str],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[RankedResult]:
    """
    Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    MMR iteratively selects documents that are:
      - RELEVANT to the query (high cosine similarity)
      - DIVERSE from already-selected documents (low inter-document similarity)

    Formula:
        MMR = λ * Sim(doc, query) - (1 - λ) * max(Sim(doc, selected_docs))

    Args:
        query_emb:       Query embedding tensor (1D).
        candidate_embs:  Candidate embeddings tensor (N x D).
        candidate_texts: List of candidate paragraph texts.
        top_k:           Number of results to return.
        lambda_param:    Trade-off between relevance (1.0) and diversity (0.0).
                         Default 0.7 = slightly favour relevance.

    Returns:
        List of RankedResult selected by MMR (ordered by selection step).
    """
    if len(candidate_texts) == 0:
        return []

    k = min(top_k, len(candidate_texts))

    # Step 1: Compute relevance scores (cosine similarity: query ↔ each candidate)
    relevance_scores = util.cos_sim(query_emb.unsqueeze(0), candidate_embs)[0]

    # Step 2: Precompute pairwise similarity matrix (candidate ↔ candidate)
    pairwise_sim = util.cos_sim(candidate_embs, candidate_embs)

    # Step 3: Iteratively select using MMR
    selected_indices: List[int] = []
    unselected_indices = list(range(len(candidate_texts)))
    results: List[RankedResult] = []

    for _ in range(k):
        best_idx = -1
        best_mmr_score = -float("inf")
        best_relevance = 0.0
        best_diversity = 0.0

        for idx in unselected_indices:
            # Relevance: how similar is this candidate to the query?
            rel = relevance_scores[idx].item()

            # Diversity: how similar is this candidate to already-selected docs?
            if selected_indices:
                max_sim_to_selected = max(
                    pairwise_sim[idx][s].item() for s in selected_indices
                )
            else:
                max_sim_to_selected = 0.0

            # MMR Formula: λ * relevance - (1-λ) * max_similarity_to_selected
            diversity = 1.0 - max_sim_to_selected
            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
                best_relevance = rel
                best_diversity = diversity

        # Select this candidate
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

        results.append(RankedResult(
            index=best_idx,
            text=candidate_texts[best_idx],
            relevance_score=best_relevance,
            diversity_score=best_diversity,
            final_score=best_mmr_score,
        ))

    return results
