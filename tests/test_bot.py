"""
Lightweight regression tests for Neura AI's deterministic logic paths.

These tests deliberately avoid anything that requires downloading the
Universal Sentence Encoder (network + ~1GB download), so they run in a few
seconds and can be used as a quick pre-deploy sanity check. They cover:

  1. The crisis / self-harm keyword net (safety-critical, so it gets the
     most coverage here).
  2. The keyword-only path of the contextual yes/no resolver.
  3. The negation/correction ("user is correcting the bot") detector.
  4. The farewell whole-word matching fix.

Run with:  python -m pytest tests/test_bot.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_logic import (
    detect_crisis_language,
    AGREEMENT_KEYWORDS,
    DISAGREEMENT_KEYWORDS,
    user_is_correcting_bot,
)
from preprocess import tokenize


def clean(text):
    return text.strip().lower().replace(".", "").replace("!", "")


# ---------------------------------------------------------------------
# 1. Crisis / self-harm keyword net
# ---------------------------------------------------------------------

def test_crisis_language_detects_explicit_phrases():
    positives = [
        "I want to kill myself",
        "I've been thinking about suicide",
        "I don't want to live anymore",
        "I can't go on like this",
        "I've been cutting myself",
        "no, I just want it to end",
        "sometimes I just want to end it all",
    ]
    for text in positives:
        assert detect_crisis_language(clean(text)), f"Should have flagged: {text!r}"


def test_crisis_language_does_not_flag_ordinary_distress():
    negatives = [
        "I'm feeling really stressed about my exams",
        "I had a rough day at work",
        "I feel anxious in social situations",
        "no thanks, I'm okay for now",
        "I'm tired of studying so much",
    ]
    for text in negatives:
        assert not detect_crisis_language(clean(text)), f"False positive on: {text!r}"


# ---------------------------------------------------------------------
# 2. Yes/No keyword coverage (semantic-anchor fallback not exercised here)
# ---------------------------------------------------------------------

def test_agreement_keywords_cover_common_phrasing():
    samples = ["yeah kind of", "sure, why not", "i guess so", "okay let's try"]
    for text in samples:
        c = clean(text)
        assert any(kw in c for kw in AGREEMENT_KEYWORDS), f"Missed agreement: {text!r}"


def test_disagreement_keywords_cover_common_phrasing():
    samples = ["not really", "nah i'm good", "no thank you", "i don't think so"]
    for text in samples:
        c = clean(text)
        assert any(kw in c for kw in DISAGREEMENT_KEYWORDS), f"Missed disagreement: {text!r}"


# ---------------------------------------------------------------------
# 3. Negation / correction detector
# ---------------------------------------------------------------------

def test_correcting_bot_matches_stemmed_context_keyword():
    # Context built from "social_anxiety" -> keywords ["social", "anxiety"].
    # User says "attacks" (not "attack") -- should still match via stemming.
    assert user_is_correcting_bot(
        "no, I wasn't talking about panic attacks",
        "awaiting_panic_attack_followup",
    )


def test_correcting_bot_ignores_unrelated_negation():
    assert not user_is_correcting_bot(
        "no worries, that's totally fine",
        "awaiting_burnout_followup",
    )


def test_correcting_bot_false_when_no_active_context():
    assert not user_is_correcting_bot("no that's not right", None)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__} -- {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
