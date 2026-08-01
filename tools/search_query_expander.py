#!/usr/bin/env python3
"""
Search Query Expander — shared query pre-processing for Hermes search tools.

Design rationale (from obsidian_search v2.2+ v3):
  All search tools default to AND semantics (>0 results only when every term
  appears). LLMs naturally pile keywords into queries like:
    "obsidian vault path location address"
  → AND over 6 terms → 0 results.

  This module provides backend-specific OR expansion + term-coverage scoring
  so multi-word queries return ranked results instead of empty sets.

Layers:
  1. Query Analysis   — determine if expansion is warranted
  2. Backend Expansion — FTS5 / regex / semantic
  3. Coverage Scoring  — re-rank results by term coverage (TF + path + bonus)

History:
  2026-05-23  Initial implementation from hermes-search-panorama design doc.
              See: 00_虚室/2026-05-23-hermes搜索全景与search_query_expander可行方案.md
"""

import math
import re
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: Query Analysis
# ═══════════════════════════════════════════════════════════════════════

# Regex special characters that indicate a crafted regex, not a keyword pile.
_REGEX_SPECIAL_CHARS = re.compile(r'[.+*?^${}\[\]()\\|]')

# Quoted phrase patterns: "exact phrase" or 'exact phrase'
_QUOTED_PHRASE_RE = re.compile(r'''(["'])(.*?)\1''')

# Boolean operator keywords (case-insensitive). Must be whole words.
_BOOLEAN_KEYWORDS_RE = re.compile(r'\b(?:OR|AND|NOT)\b', re.IGNORECASE)

# Token split: collapse whitespace, keep CJK characters as individual tokens.
# CJK range: U+4E00–U+9FFF (CJK Unified), U+3400–U+4DBF (CJK Ext-A),
#            U+F900–U+FAFF (CJK Compat), U+3000–U+303F (CJK Symbols)
_CJK_RANGE_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f]')

# Word boundary split (non-CJK text)
_WORD_SPLIT_RE = re.compile(r'[^\w]+')


def _tokenize(query: str) -> List[str]:
    """Split a query into individual tokens.

    CJK characters are treated as single-token units.
    Non-CJK text is split on word boundaries.
    """
    tokens: List[str] = []
    buf = ""
    for ch in query:
        if _CJK_RANGE_RE.match(ch):
            if buf.strip():
                tokens.extend([t for t in _WORD_SPLIT_RE.split(buf.strip()) if t])
                buf = ""
            tokens.append(ch)
        else:
            buf += ch
    if buf.strip():
        tokens.extend([t for t in _WORD_SPLIT_RE.split(buf.strip()) if t])
    return [t.lower() for t in tokens if len(t) > 0]


def _has_boolean_operators(query: str) -> bool:
    """Detect explicit boolean operators (OR/AND/NOT) in the query."""
    return bool(_BOOLEAN_KEYWORDS_RE.search(query))


def _has_quoted_phrases(query: str) -> bool:
    """Detect quoted phrases.

    Quoted phrases represent exact matches — the user deliberately grouped
    words. Expansion would destroy that intent, so they act as a guard.
    """
    return bool(_QUOTED_PHRASE_RE.search(query))


def _has_regex_special_chars(query: str) -> bool:
    """Detect regex special characters in the query.

    If present, the query is a crafted regex, NOT a keyword pile.
    Expansion would break the regex semantics.
    """
    return bool(_REGEX_SPECIAL_CHARS.search(query))


def _should_expand(query: str, min_terms: int = 3) -> bool:
    """Determine whether a query should be OR-expanded.

    Guards (return False immediately):
      - Has boolean operators → user explicitly controls logic
      - Has quoted phrases → user grouped words intentionally
      - Fewer than min_terms tokens → AND is fine (enough matches)

    The user's design note: quoted phrases are a space-handling mechanism,
    not a search syntax indicator. So they guard against expansion.
    """
    if not query or not query.strip():
        return False
    if _has_boolean_operators(query):
        return False
    if _has_quoted_phrases(query):
        return False
    tokens = _tokenize(query)
    return len(tokens) >= min_terms


# ═══════════════════════════════════════════════════════════════════════
# Layer 2: Backend-specific Expansion
# ═══════════════════════════════════════════════════════════════════════

def expand_for_fts5(query: str) -> str:
    """Expand "a b c" → "a OR b OR c" for FTS5 backends.

    FTS5 supports OR operator natively. This transform is safe because
    FTS5 treats 'OR' as a keyword, not a search term.
    """
    tokens = _tokenize(query)
    if len(tokens) <= 1:
        return query
    return " OR ".join(tokens)


def expand_for_regex(query: str) -> str:
    """Expand "a b c" → "(a|b|c)" for regex backends (rg/grep).

    Only called after _has_regex_special_chars() check returns False,
    so the tokens are all literal keyword strings.
    """
    tokens = _tokenize(query)
    if len(tokens) <= 1:
        return query
    return "(" + "|".join(tokens) + ")"


def expand_for_semantic(query: str) -> str:
    """Semantic expansion — identity function.

    Vector search uses full query for embedding. OR expansion would
    change semantics; not applicable.
    """
    return query


# ═══════════════════════════════════════════════════════════════════════
# Layer 3: Coverage Scoring
# ═══════════════════════════════════════════════════════════════════════

# Default fields to extract content from for coverage checking
_DEFAULT_CONTENT_FIELDS = ['content', 'text', 'summary']


def score_by_term_coverage(
    results: List[Dict[str, Any]],
    query: str,
    *,
    content_fields: Optional[List[str]] = None,
    name_field: Optional[str] = None,
    enable_tf: bool = True,
    name_bonus: float = 2.0,
    coverage_multiplier: float = 10.0,
) -> List[Dict[str, Any]]:
    """Re-rank results by term coverage score.

    Algorithm (generalized from obsidian v3 scoring model):

        score = Σ log₂(1 + tf_i)              ← term frequency in content
              + name_bonus × matched_in_name  ← name/path bonus
              + coverage_multiplier × (matched / total_terms)  ← coverage

    Higher score = more query terms matched in this result.

    Args:
        results: list of result dicts (each must have at least one content field)
        query: the original query string (before expansion)
        content_fields: field names to check for term hits (defaults to
                        ['content', 'text', 'summary'])
        name_field: field name with special bonus weight (e.g. 'path', 'title')
        enable_tf: whether to compute log₂ TF in scoring
        name_bonus: weight multiplier for name field matches
        coverage_multiplier: weight for (matched/total) ratio

    Returns:
        results sorted by _search_score descending (in-place modification +
        the sorted list)
    """
    if not results:
        return results

    content_fields = content_fields or _DEFAULT_CONTENT_FIELDS
    query_tokens = _tokenize(query)

    if not query_tokens:
        return results

    total_terms = len(query_tokens)

    for item in results:
        content_texts: List[str] = []
        for field in content_fields:
            val = item.get(field, "")
            if isinstance(val, str) and val:
                content_texts.append(val.lower())

        name_text = ""
        if name_field:
            name_text = item.get(name_field, "") or ""
            if isinstance(name_text, str):
                name_text = name_text.lower()

        score = 0.0
        matched = 0

        for token in query_tokens:
            token_matched = False

            # Check content fields for TF
            tf_in_item = 0
            for ct in content_texts:
                tf_in_item += ct.count(token)
            if tf_in_item > 0:
                token_matched = True
                if enable_tf:
                    score += math.log2(1 + tf_in_item)

            # Check name field for bonus
            if name_text and token in name_text:
                token_matched = True
                score += name_bonus

            if token_matched:
                matched += 1

        # Coverage bonus
        if matched > 0:
            score += coverage_multiplier * (matched / total_terms)

        item['_search_score'] = round(score, 3)

    # Sort by score descending, then preserve original order for ties
    results.sort(key=lambda x: x.get('_search_score', 0), reverse=True)
    return results


def score_session_results(
    results: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Score session_search results by term coverage.

    session_search returns results with 'snippet' (FTS5 snippet) and optionally
    'title'. We score against snippet content and title name.
    """
    return score_by_term_coverage(
        results,
        query,
        content_fields=['snippet', 'content'],
        name_field='title',
    )


def score_file_results(
    matches: List[Any],
    query: str,
) -> List[Any]:
    """Score file_search matches by term coverage.

    file_operations.py returns SearchMatch objects with .path and .content.
    We adapt to dict format for scoring and return scored dicts.

    This is designed to aggregate matches per file and score at file level,
    then re-sort individual matches by file score.
    """
    if not matches:
        return matches

    query_tokens = _tokenize(query)
    if not query_tokens:
        return matches

    total_terms = len(query_tokens)

    # Convert SearchMatch objects to dicts, score them
    scored: List[Dict[str, Any]] = []
    for m in matches:
        path = getattr(m, 'path', '')
        content = getattr(m, 'content', '')
        path_lower = path.lower() if path else ''
        content_lower = content.lower() if content else ''

        score = 0.0
        matched = 0

        for token in query_tokens:
            token_matched = False
            tf = content_lower.count(token)
            if tf > 0:
                token_matched = True
                score += math.log2(1 + tf)
            if token in path_lower:
                token_matched = True
                score += 2.0  # name_bonus
            if token_matched:
                matched += 1

        if matched > 0:
            score += 10.0 * (matched / total_terms)

        scored.append({
            '_orig': m,
            '_search_score': round(score, 3),
        })

    scored.sort(key=lambda x: x['_search_score'], reverse=True)
    return [s['_orig'] for s in scored]
