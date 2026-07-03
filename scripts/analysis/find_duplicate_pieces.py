"""
Flag candidate duplicate pieces in features/guitar_descriptors.csv that an exact
Title+Composer key misses -- e.g. the same composition matched independently
from two different source datasets (pdf vs dada_gp vs gaps) under different
titles, or the same composer spelled inconsistently across scrapes.

This is a *candidate* report for manual review, not an auto-resolver: title
similarity is a heuristic, and confirming a true duplicate (vs. two genuinely
different pieces with similar names, e.g. different movements of a suite)
needs a human/domain look, same as the Asturias case.
"""
import csv
import difflib
import re
from collections import defaultdict

CSV_PATH = "features/guitar_descriptors.csv"
COMPOSER_SIM_THRESHOLD = 0.82
CONTAINMENT_THRESHOLD = 0.9
MIN_CONTENT_TOKENS = 2  # shorter title must have >=2 non-generic tokens to trigger

# Only truly semantically-empty connectors -- deliberately NOT stripping numbers,
# roman numerals, or words like "suite"/"sonata"/"op", since those are often
# exactly what distinguishes genuinely different pieces in a numbered set
# (e.g. "Estudios Sencillos: V" vs "Estudios Sencillos: VI").
_STOPWORDS = {"the", "a", "an", "of", "from", "in", "for", "by", "to"}
_GENERIC_ONLY = {"suite", "sonata", "estudio", "estudios", "sencillos", "prelude",
                 "valse", "valses", "caprice", "caprices", "op", "opus", "no"}


def tokenize(title):
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    return {t for t in title.split() if t not in _STOPWORDS}


def containment(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / min(len(tokens1), len(tokens2))


def cluster_composers(composers):
    """Union-find style clustering of near-identical composer strings."""
    composers = sorted(composers)
    canonical = {}
    clusters = []
    for c in composers:
        match = None
        for cluster in clusters:
            if difflib.SequenceMatcher(None, c, cluster[0]).ratio() >= COMPOSER_SIM_THRESHOLD:
                match = cluster
                break
        if match:
            match.append(c)
        else:
            clusters.append([c])
    for cluster in clusters:
        rep = min(cluster, key=len)
        for c in cluster:
            canonical[c] = rep
    return canonical


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    composers = {r["Composer"].strip() for r in rows}
    canonical = cluster_composers(composers)

    merged_composer_groups = defaultdict(list)
    for rep, group in [(rep, [c for c in canonical if canonical[c] == rep]) for rep in set(canonical.values())]:
        if len(group) > 1:
            merged_composer_groups[rep] = group

    print("=== Composer spelling variants merged for grouping ===")
    for rep, group in merged_composer_groups.items():
        print(f"  {rep!r} <- {group}")
    print()

    by_composer = defaultdict(list)
    for r in rows:
        canon = canonical[r["Composer"].strip()]
        by_composer[canon].append(r)

    candidates = []
    for composer, group in by_composer.items():
        tokenized = [(r, tokenize(r["Title"])) for r in group]
        for i in range(len(tokenized)):
            for j in range(i + 1, len(tokenized)):
                r1, t1 = tokenized[i]
                r2, t2 = tokenized[j]
                shorter_len = min(len(t1), len(t2))
                shorter_content = (t1 if len(t1) <= len(t2) else t2) - _GENERIC_ONLY
                if shorter_len == 0 or len(shorter_content) < MIN_CONTENT_TOKENS:
                    continue
                score = containment(t1, t2)
                if score >= CONTAINMENT_THRESHOLD:
                    cross_source = r1["source"] != r2["source"]
                    candidates.append((cross_source, score, composer, r1, r2))

    # Cross-source matches first (same composition scraped from two datasets
    # independently is the highest-risk, most-likely-real pattern -- that's
    # exactly how the Asturias duplicate was found).
    candidates.sort(key=lambda c: (not c[0], -c[1]))

    print("=== Candidate duplicate pieces (token containment >= "
          f"{CONTAINMENT_THRESHOLD}) ===")
    for cross_source, score, composer, r1, r2 in candidates:
        tag = "CROSS-SOURCE" if cross_source else "same-source"
        print(f"[{composer}] containment={score:.2f} ({tag})")
        print(f"    '{r1['Title']}' | source={r1['source']} | diff={r1['Difficulty']} "
              f"| notes={r1['total_notes']} | tempo={r1['tempo_bpm']}")
        print(f"    '{r2['Title']}' | source={r2['source']} | diff={r2['Difficulty']} "
              f"| notes={r2['total_notes']} | tempo={r2['tempo_bpm']}")
        print()
    n_cross = sum(1 for c in candidates if c[0])
    print(f"Total candidate pairs: {len(candidates)} ({n_cross} cross-source, {len(candidates) - n_cross} same-source)")


if __name__ == "__main__":
    main()
