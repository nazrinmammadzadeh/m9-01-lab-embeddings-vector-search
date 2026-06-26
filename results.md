# Results — Search by Meaning, by Hand

Embedding backend: `sentence-transformers/all-MiniLM-L6-v2` (keyless local fallback;
identical model used for passages and queries). Cosine similarity computed by hand
with NumPy. Run with `python embeddings_search.py`.

## Top-3 per query

### "my laptop won't switch on"
| rank | score | id | source | word overlap |
|------|-------|----|--------|--------------|
| 1 | 0.437 | kb-02 | handbook.md | only "on" / "won't" |
| 2 | 0.177 | kb-07 | it.md | none |
| 3 | 0.158 | kb-09 | it.md | none |

Best match **kb-02** ("power up a device that won't turn on") shares no meaningful
words with "laptop"/"switch on" — matched purely on meaning.

### "how do I stop being billed every month?"
| rank | score | id | source | word overlap |
|------|-------|----|--------|--------------|
| 1 | 0.541 | kb-05 | policy.md | **none** |
| 2 | 0.304 | kb-06 | policy.md | none |
| 3 | 0.205 | kb-10 | facilities.md | only "every" |

Best match **kb-05** ("cancel your subscription … billing period") shares **zero**
words with the query, yet is the clear winner.

### "access denied error when saving a file"
| rank | score | id | source | word overlap |
|------|-------|----|--------|--------------|
| 1 | 0.544 | kb-08 | it.md | "access", "denied", "error" |
| 2 | 0.121 | kb-09 | it.md | none |
| 3 | 0.086 | kb-07 | it.md | only "a" |

Here there *is* overlap, and the right passage (**kb-08**, error 0x80070005) also wins.

### "where do I leave my car in the evening?"
| rank | score | id | source | word overlap |
|------|-------|----|--------|--------------|
| 1 | 0.326 | kb-01 | handbook.md | only "in" |
| 2 | 0.218 | kb-03 | handbook.md | "in", "leave", "the" |
| 3 | 0.183 | kb-10 | facilities.md | "evening", "in", "the" |

Best match **kb-01** ("park in lot B after 6pm") shares no content words with
"leave my car in the evening" — "park" ≈ "leave my car", "after 6pm" ≈ "evening".
Note kb-03 shares *more* literal words ("leave") but is correctly ranked lower,
because its "leave" means annual leave, not parking — meaning beats overlap.

## Reflection (step 4)

For three of the four intended queries the top passage shares **few or no words**
with the query, yet it is ranked first. That is the whole point: the embedding
captured **meaning**, not surface vocabulary. Synonyms and paraphrases
("switch on" ≈ "turn on", "billed monthly" ≈ "subscription billing period",
"leave my car" ≈ "park") land near each other in vector space, so a hand-written
dot product retrieves the right passage even with zero shared words — something a
keyword search would miss entirely.

## Optional stretch — uncovered query

### "what's the wifi password?"
| rank | score | id | source |
|------|-------|----|--------|
| 1 | 0.319 | kb-07 | it.md |
| 2 | 0.123 | kb-09 | it.md |
| 3 | 0.121 | kb-02 | handbook.md |

The KB has nothing about wifi, so the best score (**0.319**) is materially lower
than the answerable queries' top scores (0.44–0.54). The model latches onto the
word "password" (kb-07 is about resetting a *login* password), but it's a weak,
off-topic match. A **similarity threshold** exploits exactly this gap: if even the
best passage scores below a chosen cutoff (say ~0.40 here), the system should
respond *"I don't have an answer for this"* rather than returning the least-bad
but irrelevant passage.
