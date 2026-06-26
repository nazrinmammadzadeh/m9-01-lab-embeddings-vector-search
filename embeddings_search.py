"""
Lab | Search by Meaning, by Hand
================================

Turn a small knowledge base into embeddings, turn a few questions into
embeddings, and find the best-matching passages by computing cosine
similarity *by hand* with NumPy -- no vector store, no built-in search.

Embeddings come from Gemini's `gemini-embedding-001` when GOOGLE_API_KEY is
set; otherwise we fall back to a local, keyless `sentence-transformers`
model. The SAME model is used for both passages and queries either way.

Run:  python embeddings_search.py
"""

import json
import os
from pathlib import Path

import numpy as np

KB_PATH = Path(__file__).with_name("knowledge_base.json")

TEST_QUERIES = [
    "my laptop won't switch on",
    "how do I stop being billed every month?",
    "access denied error when saving a file",
    "where do I leave my car in the evening?",
    # Optional stretch: a query the KB does NOT cover.
    "what's the wifi password?",
]


# --------------------------------------------------------------------------- #
# Embedding backend: Gemini if a key is present, else local sentence-transformers
# --------------------------------------------------------------------------- #
def make_embedder():
    """Return (embed_fn, backend_name). embed_fn maps list[str] -> np.ndarray."""
    api_key = os.environ.get("GOOGLE_API_KEY")

    if api_key:
        from google import genai

        client = genai.Client(api_key=api_key)

        def embed(texts):
            resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
            )
            return np.array([e.values for e in resp.embeddings], dtype=np.float32)

        return embed, "gemini-embedding-001"

    # Keyless local fallback.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return np.asarray(
            model.encode(texts, normalize_embeddings=False), dtype=np.float32
        )

    return embed, "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Cosine similarity, written out by hand (no library search function).
# --------------------------------------------------------------------------- #
def cosine_similarity(query_vec, matrix):
    """Cosine sim between one vector and every row of `matrix`.

    cos(a, b) = (a . b) / (||a|| * ||b||)
    """
    dots = matrix @ query_vec                       # numerator: dot products
    matrix_norms = np.linalg.norm(matrix, axis=1)   # ||b|| for each passage
    query_norm = np.linalg.norm(query_vec)          # ||a||
    return dots / (matrix_norms * query_norm + 1e-10)


def words(text):
    return {w.strip(".,!?'\"()").lower() for w in text.split()}


def main():
    kb = json.loads(KB_PATH.read_text())
    embed, backend = make_embedder()
    print(f"Embedding backend: {backend}\n")

    # 1. Embed every passage ONCE, keeping id/source alongside the vector.
    passage_vectors = embed([entry["text"] for entry in kb])

    # 2-3. For each query: embed it, score every passage by hand, show top 3.
    for query in TEST_QUERIES:
        query_vec = embed([query])[0]
        scores = cosine_similarity(query_vec, passage_vectors)
        top3 = np.argsort(scores)[::-1][:3]

        print(f"QUERY: {query!r}")
        query_words = words(query)
        for rank, idx in enumerate(top3, start=1):
            entry = kb[idx]
            shared = query_words & words(entry["text"])
            shared_note = f"shared words: {sorted(shared)}" if shared else "NO shared words"
            print(
                f"  {rank}. [{scores[idx]:.3f}] {entry['id']} ({entry['source']}) "
                f"-- {shared_note}"
            )
            print(f"       {entry['text'][:90]}...")
        print()


if __name__ == "__main__":
    main()


# ===========================================================================
# Reflection (step 4)
# ===========================================================================
# For each of the four intended queries, the best-matching passage shares
# FEW or NO literal words with the query, yet it is still ranked first:
#
#   * "my laptop won't switch on"            -> kb-02 "power up a device that
#     won't turn on..."  (no overlap on "laptop"/"switch on"; matched on meaning)
#   * "how do I stop being billed every month?" -> kb-05 "cancel your
#     subscription... billing period"  (no shared words like "stop"/"billed")
#   * "access denied error when saving a file"  -> kb-08 "error code 0x80070005
#     means 'access denied'... write permission to the target folder"
#   * "where do I leave my car in the evening?" -> kb-01 "Employees may park in
#     lot B after 6pm..."  (no overlap on "leave"/"car"/"evening")
#
# This shows the embedding captured MEANING, not surface word overlap: synonyms
# and paraphrases ("switch on" ~ "turn on", "billed monthly" ~ "subscription
# billing period", "leave my car" ~ "park") land near each other in vector
# space. A plain keyword search would miss every one of these.
#
# Optional stretch -- the uncovered query "what's the wifi password?":
# its top score is noticeably LOWER than the answerable queries' top scores,
# because nothing in the KB is genuinely about it. That gap is exactly what a
# SIMILARITY THRESHOLD exploits: if even the best passage scores below a chosen
# cutoff, the system should say "I don't have an answer for this" instead of
# returning the least-bad (but irrelevant) passage. (See the printed numbers.)
