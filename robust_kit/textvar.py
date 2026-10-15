"""Text variations: Family A/B perturbations and rule-based normalizers for Hinglish.

Perturbation Families:
- Family A (Respelling): elongation, aa/a, ee/i, oo/u, ph/f, v/w. (Usable for training augmentation)
- Family B (Shorthand/Typos): vowel dropping, keyboard-neighbour typos, adjacent swaps, consonant doubling.
  (HELD OUT: never used for training augmentation or normalizer design)

Normalizers:
- norm_light: repeated char collapse, standard respelling canonicalization (ph->f, w->v, ee->i, oo->u, aa->a)
- norm_skeleton: lossy consonant skeleton reduction (ablation)
"""

from __future__ import annotations

import random
import re
from typing import Callable, Sequence

# Protected negation words to preserve sentiment polarity
NEGATION_WORDS = {
    "nahi", "nahin", "nhi", "nhn", "na", "mat", "ni", "ney",
    "not", "no", "never", "dont", "don't", "cant", "can't",
    "wont", "won't", "neither", "nor", "bina", "kabhi"
}

# QWERTY keyboard adjacent map for realistic typos
KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "ersfxc", "e": "wsdr",
    "f": "rtgdvc", "g": "tyhfvb", "h": "yujgbn", "i": "ujko", "j": "uikhmn",
    "k": "ijolm", "l": "okp", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "wedxza", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx"
}


# ==============================================================================
# Normalizers
# ==============================================================================

def normalize_light(text: str) -> str:
    """Light rule-based normalizer for Romanized Hinglish.
    
    1. Lowercase and normalize repeated characters (3+ identical chars reduced to 1 or 2).
    2. Canonicalize vowel and consonant respelling digraphs:
       - 'ph' -> 'f'
       - 'w' -> 'v'
       - 'aa' -> 'a'
       - 'ee' -> 'i'
       - 'oo' -> 'u'
    3. Normalize whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return "" if not isinstance(text, str) else text
    
    # 1. Lowercase
    s = text.lower()
    
    # 2. Collapse 3+ repeated characters to 1 (e.g. bohooooot -> bohot, haaaan -> han)
    s = re.sub(r"(.)\1{2,}", r"\1", s)
    
    # 3. Canonicalize digraphs / spelling variants word-internally
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"w", "v", s)
    s = re.sub(r"aa", "a", s)
    s = re.sub(r"ee", "i", s)
    s = re.sub(r"oo", "u", s)
    
    # 4. Collapse leftover double characters that are not standard (leave ss, ll, etc. or single)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_skeleton(text: str) -> str:
    """Lossy consonant-skeleton normalizer (ablation).
    
    Reduces words to their consonant skeletons while preserving word-initial vowels
    and protected negations.
    """
    if not isinstance(text, str) or not text.strip():
        return "" if not isinstance(text, str) else text
    
    # First apply light normalization
    s = normalize_light(text)
    tokens = s.split()
    skeleton_tokens: list[str] = []
    
    for token in tokens:
        # Keep punctuation or non-alpha tokens intact
        if not token.isalpha() or token in NEGATION_WORDS or len(token) <= 2:
            skeleton_tokens.append(token)
            continue
        
        # Keep initial letter, strip internal/trailing vowels (a, e, i, o, u)
        first_char = token[0]
        rest = token[1:]
        consonants_rest = re.sub(r"[aeiou]", "", rest)
        
        # Ensure we do not produce empty or degenerate single-char token if rest had no consonants
        token_skel = first_char + consonants_rest
        if len(token_skel) < 2 and len(token) >= 2:
            token_skel = token[:2]
        skeleton_tokens.append(token_skel)
        
    return " ".join(skeleton_tokens)


# ==============================================================================
# Family A Perturbations: Respelling (Train-time Augmentation candidate)
# ==============================================================================

def _rule_elongation(word: str, rng: random.Random) -> str:
    """Elongate a vowel or liquid consonant by repeating 2-4 times."""
    candidates = [i for i, c in enumerate(word) if c in "aeiouy"]
    if not candidates:
        candidates = list(range(len(word)))
    idx = rng.choice(candidates)
    repeat_count = rng.randint(2, 4)
    return word[:idx] + word[idx] * repeat_count + word[idx + 1:]


def _rule_vowel_respell(word: str, rng: random.Random) -> str:
    """Substitute aa <-> a, ee <-> i, oo <-> u."""
    options = []
    if "aa" in word:
        options.append(("aa", "a"))
    elif "a" in word:
        options.append(("a", "aa"))
    if "ee" in word:
        options.append(("ee", "i"))
    elif "i" in word:
        options.append(("i", "ee"))
    if "oo" in word:
        options.append(("oo", "u"))
    elif "u" in word:
        options.append(("u", "oo"))
        
    if options:
        src, tgt = rng.choice(options)
        # Replace only one occurrence
        return word.replace(src, tgt, 1)
    return _rule_elongation(word, rng)


def _rule_consonant_respell(word: str, rng: random.Random) -> str:
    """Substitute ph <-> f, v <-> w."""
    options = []
    if "ph" in word:
        options.append(("ph", "f"))
    elif "f" in word:
        options.append(("f", "ph"))
    if "v" in word:
        options.append(("v", "w"))
    elif "w" in word:
        options.append(("w", "v"))
        
    if options:
        src, tgt = rng.choice(options)
        return word.replace(src, tgt, 1)
    return _rule_vowel_respell(word, rng)


def perturb_token_family_a(word: str, rng: random.Random) -> str:
    """Apply one Family A perturbation rule to a single token."""
    if len(word) <= 2 or word.lower() in NEGATION_WORDS or not word.isalpha():
        return word
    rule = rng.choice([_rule_elongation, _rule_vowel_respell, _rule_consonant_respell])
    result = rule(word, rng)
    return result if result else word


# ==============================================================================
# Family B Perturbations: Shorthand & Typos (HELD OUT - Evaluation only)
# ==============================================================================

def _rule_vowel_drop(word: str, rng: random.Random) -> str:
    """Drop an internal vowel (e.g., bahut -> bht, samajh -> smjh)."""
    if len(word) <= 3:
        return word
    # internal vowels (exclude first character to preserve recognizability)
    internal_vowels = [i for i in range(1, len(word)) if word[i] in "aeiou"]
    if internal_vowels:
        idx = rng.choice(internal_vowels)
        res = word[:idx] + word[idx + 1:]
        # Ensure at least 2 chars remain
        if len(res) >= 2:
            return res
    return word


def _rule_keyboard_typo(word: str, rng: random.Random) -> str:
    """Substitute a letter with an adjacent QWERTY keyboard neighbour."""
    indices = [i for i, c in enumerate(word) if c.lower() in KEYBOARD_NEIGHBORS]
    if not indices:
        return word
    idx = rng.choice(indices)
    char = word[idx].lower()
    neighbor = rng.choice(KEYBOARD_NEIGHBORS[char])
    if word[idx].isupper():
        neighbor = neighbor.upper()
    return word[:idx] + neighbor + word[idx + 1:]


def _rule_adjacent_swap(word: str, rng: random.Random) -> str:
    """Swap two adjacent letters (e.g. yeh -> yhe)."""
    if len(word) <= 2:
        return word
    idx = rng.randint(0, len(word) - 2)
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def _rule_consonant_double(word: str, rng: random.Random) -> str:
    """Double a consonant (e.g. hum -> humm, sach -> sacch)."""
    consonants = [i for i, c in enumerate(word) if c.isalpha() and c.lower() not in "aeiou"]
    if not consonants:
        return word
    idx = rng.choice(consonants)
    return word[:idx] + word[idx] + word[idx:]


def perturb_token_family_b(word: str, rng: random.Random) -> str:
    """Apply one Family B perturbation rule to a single token."""
    if len(word) <= 2 or word.lower() in NEGATION_WORDS or not word.isalpha():
        return word
    rule = rng.choice([_rule_vowel_drop, _rule_keyboard_typo, _rule_adjacent_swap, _rule_consonant_double])
    result = rule(word, rng)
    return result if result else word


# ==============================================================================
# Sentence-Level Perturbation APIs
# ==============================================================================

def perturb_sentence(
    text: str,
    family: str,
    p: float = 0.3,
    seed: int | None = None
) -> str:
    """Perturb tokens in text with probability p using specified family (A or B).
    
    If p <= 0, returns the text strictly unchanged (no-op sanity check).
    """
    if not isinstance(text, str) or not text.strip() or p <= 0.0:
        return text
    
    rng = random.Random(seed)
    tokens = text.split()
    new_tokens: list[str] = []
    
    perturb_fn = perturb_token_family_a if family.upper() == "A" else perturb_token_family_b
    
    for token in tokens:
        # Protect punctuation, urls, mentions, negations, or non-alpha tokens
        if (
            token.lower() in NEGATION_WORDS
            or not token.isalpha()
            or len(token) <= 2
            or token.startswith("@")
            or token.startswith("#")
            or token.startswith("http")
        ):
            new_tokens.append(token)
            continue
            
        if rng.random() < p:
            new_token = perturb_fn(token, rng)
            new_tokens.append(new_token)
        else:
            new_tokens.append(token)
            
    return " ".join(new_tokens)


def augment_dataset(
    texts: Sequence[str],
    labels: Sequence[str],
    p: float = 0.3,
    seed: int = 42
) -> tuple[list[str], list[str]]:
    """Generate train-time augmented dataset using Family A ONLY.
    
    Duplicates each training instance with a perturbed version under Family A.
    """
    rng = random.Random(seed)
    aug_texts: list[str] = []
    aug_labels: list[str] = []
    
    for text, label in zip(texts, labels):
        aug_texts.append(text)
        aug_labels.append(label)
        
        # Perturbed copy with unique sub-seed
        instance_seed = rng.randint(0, 10_000_000)
        perturbed = perturb_sentence(text, family="A", p=p, seed=instance_seed)
        if perturbed != text:
            aug_texts.append(perturbed)
            aug_labels.append(label)
            
    return aug_texts, aug_labels
