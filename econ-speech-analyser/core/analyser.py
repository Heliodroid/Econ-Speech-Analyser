"""
core/analyser.py
----------------
Heart of the project. Takes raw text, returns structured tone analysis.
No UI here — just pure logic. Other modules import from this.
"""

import re
import nltk
import textstat
import pandas as pd
from collections import Counter
from datetime import datetime
from pathlib import Path

# Download NLTK data on first run (silent after that)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# ─── Domain lexicons ──────────────────────────────────────────────────────────
# WHY hand-crafted and not a generic sentiment library?
# Because "easing" is positive in everyday English but DOVISH in central bank
# speak. "Tight" labor market is good news, but "tight policy" is hawkish.
# Domain-specific dictionaries beat generic tools for specialised corpora.

HAWKISH_WORDS = {
    "inflation", "hike", "hikes", "tighten", "tightening", "restrict",
    "restrictive", "aggressive", "combat", "overheat", "overheating",
    "pressure", "pressures", "surge", "surging", "persistent", "persistently",
    "above", "target", "vigilant", "vigilance", "expeditiously", "brisk",
    "resolve", "resolved", "unacceptable", "mandate", "forceful", "forcefully",
    "reduce", "reducing", "runaway", "elevated", "broad-based", "entrenched",
    "credibility", "anchor", "anchoring", "normalise", "normalising",
    "withdrawal", "contracted", "contraction"
}

DOVISH_WORDS = {
    "accommodate", "accommodative", "ease", "easing", "support", "supportive",
    "pause", "pausing", "patient", "patience", "gradual", "gradually",
    "below", "flexible", "flexibility", "monitor", "monitoring", "transitory",
    "recover", "recovery", "stimulus", "unemployment", "growth", "slowdown",
    "slow", "sluggish", "weak", "weakness", "downside", "risks", "uncertain",
    "uncertainty", "cautious", "caution", "data-dependent", "hold", "holding",
    "cut", "cuts", "lower", "lowering", "expansion", "expansionary"
}

STOP_WORDS = set(stopwords.words("english"))

# ─── CSV log path (relative to project root) ─────────────────────────────────
LOG_PATH = Path(__file__).parent.parent / "data" / "results.csv"


def clean_and_tokenise(text: str) -> list[str]:
    """
    Lowercase, tokenise, keep only alphabetic words.
    TODDLER EXPLANATION: Take a wall of text. Chop it into words.
    Throw away numbers and punctuation. Make everything lowercase
    so 'Inflation' and 'inflation' count as the same word.
    """
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Drop words so common they carry zero information.
    TODDLER EXPLANATION: 'The', 'is', 'and', 'of' appear in EVERY text.
    They're like background noise. Remove them so we only hear the signal.
    """
    return [w for w in tokens if w not in STOP_WORDS]


def compute_tone_score(filtered_tokens: list[str]) -> dict:
    """
    Match tokens against hawkish/dovish lexicons.
    Normalise by total filtered word count so long speeches don't
    automatically score higher than short ones.

    TODDLER EXPLANATION: Count how many 'angry inflation fighter' words
    vs 'calm growth supporter' words exist. Divide by total words
    so a 10-word speech and a 1000-word speech are comparable.

    Returns score > 0  → hawkish lean
            score < 0  → dovish lean
            score ≈ 0  → neutral / balanced
    """
    hawk_hits = [w for w in filtered_tokens if w in HAWKISH_WORDS]
    dove_hits = [w for w in filtered_tokens if w in DOVISH_WORDS]
    total = len(filtered_tokens) or 1  # avoid division by zero

    score = (len(hawk_hits) - len(dove_hits)) / total

    if score > 0.015:
        label = "Hawkish"
    elif score < -0.010:
        label = "Dovish"
    else:
        label = "Neutral"

    return {
        "score": round(score, 4),
        "label": label,
        "hawk_hits": hawk_hits,
        "dove_hits": dove_hits,
        "hawk_count": len(hawk_hits),
        "dove_count": len(dove_hits),
    }


def readability_metrics(text: str) -> dict:
    """
    Flesch Reading Ease: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    Score 0-30  → very hard (academic/legal)
    Score 30-50 → difficult (Fed minutes live here ~35)
    Score 60-70 → standard (newspaper)
    Score 70+   → easy

    TODDLER EXPLANATION: Long sentences with long words = hard to read.
    Short sentences with simple words = easy. Fed speeches score ~35.
    If YOU score lower than the Fed, write shorter sentences!
    """
    return {
        "flesch_score": round(textstat.flesch_reading_ease(text), 1),
        "grade_level": round(textstat.flesch_kincaid_grade(text), 1),
        "sentence_count": textstat.sentence_count(text),
        "avg_words_per_sentence": round(
            len(text.split()) / max(textstat.sentence_count(text), 1), 1
        ),
    }


def top_keywords(filtered_tokens: list[str], n: int = 10) -> list[tuple]:
    """
    Frequency distribution of meaningful words after stopword removal.
    TODDLER EXPLANATION: After throwing away boring words, count what's left.
    The most frequent words = what the speaker ACTUALLY cares about.
    """
    return Counter(filtered_tokens).most_common(n)


def analyse(text: str, source: str = "unknown", save: bool = True) -> dict:
    """
    Master function. Feed it text, get back a full analysis dict.
    Optionally logs to CSV for the time-series research module.

    TODDLER EXPLANATION: This is the factory. Raw text goes in one end.
    Numbers, labels, and insights come out the other end.
    """
    tokens = clean_and_tokenise(text)
    filtered = remove_stopwords(tokens)

    tone = compute_tone_score(filtered)
    readability = readability_metrics(text)
    keywords = top_keywords(filtered)

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": source,
        "word_count": len(tokens),
        **tone,
        **readability,
        "top_keywords": keywords,
    }

    if save:
        _log_to_csv(result)

    return result


def _log_to_csv(result: dict):
    """Append one row to the running CSV log. Creates file if missing."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    row = {k: v for k, v in result.items() if k not in ("hawk_hits", "dove_hits", "top_keywords")}
    row["top_keywords"] = str(result["top_keywords"])
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"[log] Saved to {LOG_PATH}")


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Econ Speech Tone Analyser ===\n")
    text = input("Paste speech text (then press Enter twice):\n")
    source = input("Source label (e.g. 'Fed_2022_Nov'): ").strip() or "unknown"

    r = analyse(text, source=source)

    print(f"\n{'─'*40}")
    print(f"Tone:        {r['label']}  (score: {r['score']})")
    print(f"Hawkish:     {r['hawk_count']} hits → {r['hawk_hits'][:5]}")
    print(f"Dovish:      {r['dove_count']} hits → {r['dove_hits'][:5]}")
    print(f"Top words:   {[w for w,_ in r['top_keywords'][:5]]}")
    print(f"Readability: Flesch {r['flesch_score']} | Grade {r['grade_level']}")
    print(f"{'─'*40}\n")
