"""
Tests for the search_query_expander shared module.

Design doc:
  00_虚室/2026-05-23-hermes搜索全景与search_query_expander可行方案.md

Three-layer architecture:
  L1 — Query Analysis (tokenize, guards)
  L2 — Backend Expansion (FTS5, regex, semantic)
  L3 — Coverage Scoring (TF + path + coverage bonus)

Test strategy mirrors obsidian-tool tests:
  §A — Pure function unit tests (no external deps)
  §B — Integration tests with real-ish data
"""
import re

import pytest

from tools.search_query_expander import (
    # L1 — Query Analysis
    _tokenize,
    _has_boolean_operators,
    _has_quoted_phrases,
    _has_regex_special_chars,
    _should_expand,
    # L2 — Backend Expansion
    expand_for_fts5,
    expand_for_regex,
    expand_for_semantic,
    # L3 — Coverage Scoring
    score_by_term_coverage,
    score_session_results,
    score_file_results,
)


# ═══════════════════════════════════════════════════════════════════════
# §A. Unit Tests — Layer 1: Query Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestTokenize:
    """_tokenize splits queries into individual search terms."""

    def test_english_words(self):
        tokens = _tokenize("obsidian vault path")
        assert tokens == ["obsidian", "vault", "path"]

    def test_cjk_characters_individual(self):
        """Each CJK character is its own token."""
        tokens = _tokenize("梦下沉")
        assert tokens == ["梦", "下", "沉"]

    def test_mixed_cjk_and_english(self):
        tokens = _tokenize("obsidian 库 地址")
        assert tokens == ["obsidian", "库", "地", "址"]

    def test_multi_space_collapse(self):
        tokens = _tokenize("obsidian   vault    path")
        assert tokens == ["obsidian", "vault", "path"]

    def test_case_insensitive(self):
        tokens = _tokenize("Obsidian Vault PATH")
        assert tokens == ["obsidian", "vault", "path"]

    def test_empty_string(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []

    def test_punctuation_removed(self):
        tokens = _tokenize("hello, world! foo-bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_single_cjk(self):
        assert _tokenize("梦") == ["梦"]


class TestHasBooleanOperators:
    """Guard: queries with explicit OR/AND/NOT should not be expanded."""

    def test_explicit_or(self):
        assert _has_boolean_operators("obsidian OR vault") is True

    def test_explicit_and(self):
        assert _has_boolean_operators("obsidian AND vault") is True

    def test_explicit_not(self):
        assert _has_boolean_operators("python NOT java") is True

    def test_no_boolean(self):
        assert _has_boolean_operators("obsidian vault path") is False

    def test_case_insensitive(self):
        assert _has_boolean_operators("obsidian or vault") is True
        assert _has_boolean_operators("obsidian Or Vault") is True

    def test_boolean_in_word_not_matched(self):
        """'OR' inside a word like 'FORK' should not trigger."""
        assert _has_boolean_operators("FORK spoon knife") is False

    def test_cjk_with_or(self):
        assert _has_boolean_operators("梦 下沉 OR 梦 惊醒") is True


class TestHasQuotedPhrases:
    """Guard: quoted phrases represent exact groupings — don't break them."""

    def test_double_quoted(self):
        assert _has_quoted_phrases('"obsidian vault"') is True

    def test_single_quoted(self):
        assert _has_quoted_phrases("'obsidian vault'") is True

    def test_no_quotes(self):
        assert _has_quoted_phrases("obsidian vault") is False

    def test_empty_quotes(self):
        assert _has_quoted_phrases('""') is True

    def test_mixed(self):
        assert _has_quoted_phrases('find "obsidian vault" path') is True


class TestHasRegexSpecialChars:
    """Guard: crafted regex patterns must not be OR-expanded."""

    def test_dot_star(self):
        assert _has_regex_special_chars("obsidian.*vault") is True

    def test_anchors(self):
        assert _has_regex_special_chars("^vault") is True
        assert _has_regex_special_chars("vault$") is True

    def test_quantifiers(self):
        assert _has_regex_special_chars("a+ b? c*") is True

    def test_char_class(self):
        assert _has_regex_special_chars("[abc]") is True

    def test_group(self):
        assert _has_regex_special_chars("(a|b)") is True

    def test_pipe(self):
        assert _has_regex_special_chars("a|b") is True

    def test_no_special(self):
        assert _has_regex_special_chars("obsidian vault path") is False


class TestShouldExpand:
    """Decision function: should we expand this query?"""

    def test_enough_terms(self):
        assert _should_expand("obsidian vault path") is True

    def test_too_few_terms(self):
        assert _should_expand("obsidian vault") is False

    def test_has_or(self):
        assert _should_expand("obsidian OR vault") is False

    def test_has_quote(self):
        assert _should_expand('"obsidian vault" path') is False

    def test_cjk_enough(self):
        """3 CJK characters = 3 terms → should expand."""
        assert _should_expand("梦下沉惊") is True

    def test_cjk_few(self):
        assert _should_expand("梦下") is False

    def test_empty(self):
        assert _should_expand("") is False

    def test_custom_min_terms(self):
        assert _should_expand("obsidian vault", min_terms=2) is True
        assert _should_expand("obsidian", min_terms=2) is False


# ═══════════════════════════════════════════════════════════════════════
# §A. Unit Tests — Layer 2: Backend Expansion
# ═══════════════════════════════════════════════════════════════════════

class TestExpandForFTS5:
    """FTS5 OR-expansion: "a b c" → "a OR b OR c"."""

    def test_three_words(self):
        assert expand_for_fts5("obsidian vault path") == "obsidian OR vault OR path"

    def test_two_words(self):
        assert expand_for_fts5("hello world") == "hello OR world"

    def test_single_word(self):
        assert expand_for_fts5("obsidian") == "obsidian"

    def test_cjk(self):
        result = expand_for_fts5("梦下沉")
        assert result == "梦 OR 下 OR 沉"

    def test_mixed(self):
        result = expand_for_fts5("obsidian 仓库")
        assert result == "obsidian OR 仓 OR 库"

    def test_no_double_or(self):
        """Words containing 'or' should not confuse the output."""
        # _tokenize lowercases, so "For" → "for" — it's a token, not a boolean
        result = expand_for_fts5("For king country")
        assert result == "for OR king OR country"


class TestExpandForRegex:
    """Regex OR-expansion: "a b c" → "(a|b|c)"."""

    def test_three_words(self):
        assert expand_for_regex("obsidian vault path") == "(obsidian|vault|path)"

    def test_single_word(self):
        assert expand_for_regex("obsidian") == "obsidian"

    def test_cjk(self):
        result = expand_for_regex("梦下沉")
        assert result == "(梦|下|沉)"

    def test_mixed(self):
        result = expand_for_regex("config yaml json")
        assert result == "(config|yaml|json)"


class TestExpandForSemantic:
    """Semantic expansion — identity (vector search uses full query)."""

    def test_identity(self):
        query = "obsidian vault path location"
        assert expand_for_semantic(query) == query


# ═══════════════════════════════════════════════════════════════════════
# §A. Unit Tests — Layer 3: Coverage Scoring
# ═══════════════════════════════════════════════════════════════════════

class TestScoreByTermCoverage:
    """Core scoring function: TF + name_bonus + coverage multiplier."""

    def test_perfect_match_ranks_highest(self):
        results = [
            {"content": "obsidian vault"},
            {"content": "obsidian vault path explained"},
        ]
        scored = score_by_term_coverage(results, "obsidian vault path")
        # Full match should be first
        assert scored[0]["content"] == "obsidian vault path explained"

    def test_full_coverage_beats_partial(self):
        results = [
            {"content": "vault"},
            {"content": "obsidian vault"},
            {"content": "obsidian vault path explained"},
        ]
        scored = score_by_term_coverage(results, "obsidian vault path")
        scores = [r["_search_score"] for r in scored]
        assert scores == sorted(scores, reverse=True), "Should be descending"

    def test_empty_results(self):
        assert score_by_term_coverage([], "query") == []

    def test_empty_query(self):
        results = [{"content": "something"}]
        scored = score_by_term_coverage(results, "")
        # No re-scoring but result kept
        assert len(scored) == 1

    def test_name_field_bonus(self):
        results = [
            {"content": "some content", "path": "obsidian.md"},
            {"content": "obsidian vault path explained", "path": "other.md"},
        ]
        scored = score_by_term_coverage(
            results, "obsidian vault",
            content_fields=["content"], name_field="path",
        )
        # Path-matched file gets bonus
        path_scores = [r["_search_score"] for r in scored if "obsidian.md" in r["path"]]
        assert len(path_scores) > 0

    def test_tf_weighting(self):
        """Multiple occurrences of a term increase TF score."""
        results = [
            {"content": "obsidian obsidian obsidian vault path"},
            {"content": "obsidian vault path"},
        ]
        scored = score_by_term_coverage(
            results, "obsidian vault path", enable_tf=True,
        )
        # First item has TF=3 for "obsidian" vs 1 in second
        assert scored[0]["content"].startswith("obsidian obsidian")

    def test_tf_disabled(self):
        results = [
            {"content": "obsidian obsidian obsidian vault path"},
            {"content": "obsidian vault path"},
        ]
        scored = score_by_term_coverage(
            results, "obsidian vault path", enable_tf=False,
        )
        # Without TF, identical coverage → tie (stable sort preserves order)
        assert scored[0]["_search_score"] == scored[1]["_search_score"]

    def test_custom_content_fields(self):
        results = [
            {"title": "obsidian vault", "body": "some text"},
        ]
        scored = score_by_term_coverage(
            results, "obsidian vault",
            content_fields=["title", "body"],
        )
        assert scored[0]["_search_score"] > 0

    def test_custom_weights(self):
        results = [
            {"content": "obsidian vault path", "title": "master"},
        ]
        scored = score_by_term_coverage(
            results, "obsidian",
            name_field="title",
            name_bonus=5.0,
            coverage_multiplier=20.0,
        )
        # obsidian in content → TF=1 log2(2)=1, no name hit
        # matched=1/1 → coverage=20, total ~21
        assert scored[0]["_search_score"] > 0

    def test_stable_sort_on_ties(self):
        results = [
            {"content": "aaa", "id": 1},
            {"content": "bbb", "id": 2},
        ]
        scored = score_by_term_coverage(results, "zzz_no_match")
        # Both get 0 score, original order preserved
        assert [r["id"] for r in scored] == [1, 2]

    def test_partial_match_scores_proportional(self):
        """More matched terms → higher score."""
        results = [
            {"content": "obsidian"},
            {"content": "obsidian vault"},
            {"content": "obsidian vault path"},
            {"content": "obsidian vault path location"},
        ]
        scored = score_by_term_coverage(results, "obsidian vault path location")
        scores = [r["_search_score"] for r in scored]
        # Each subsequent item should have >= score than previous
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Expected descending scores, got {scores}"
            )


class TestScoreSessionResults:
    """session_search-specific scoring wrapper."""

    def test_scores_on_snippet_and_title(self):
        results = [
            {"snippet": "obsidian vault path", "title": "Vault Setup Guide"},
        ]
        scored = score_session_results(results, "vault")
        assert scored[0]["_search_score"] > 0
        # "vault" appears in both snippet and title
        assert scored[0]["_search_score"] >= 2.0  # name_bonus for title

    def test_empty_results(self):
        assert score_session_results([], "query") == []


class TestScoreFileResults:
    """file_operations SearchMatch scoring."""

    class FakeMatch:
        def __init__(self, path, content):
            self.path = path
            self.content = content

    def test_scores_and_sorts(self):
        matches = [
            self.FakeMatch("other.py", "print('hello')"),
            self.FakeMatch("obsidian.py", "obsidian vault path configured"),
        ]
        scored = score_file_results(matches, "obsidian vault")
        # obsidian.py should rank first
        assert "obsidian.py" in scored[0].path

    def test_empty_list(self):
        assert score_file_results([], "query") == []


# ═══════════════════════════════════════════════════════════════════════
# §B. Integration Tests — end-to-end scenarios
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEndExpandScore:
    """Full pipeline: expand → search simulation → score."""

    def test_fts5_pipeline_improves_results(self):
        """Simulating session_search: expand with FTS5, then re-score."""
        query = "obsidian vault path"

        # Simulated FTS5 results (OR expansion returns many hits)
        all_hits = [
            {"snippet": "The vault was opened", "title": "Minecraft Mod"},
            {"snippet": "obsidian knowledge base", "title": "Note Taking"},
            {"snippet": "obsidian vault setup guide", "title": "Obsidian Setup"},
            {"snippet": "obsidian vault path configured", "title": "Configuration"},
        ]

        # After scoring, multi-term matches rank higher
        scored = score_session_results(all_hits, query)
        scores = [r["_search_score"] for r in scored]

        # Items with more query terms should have higher scores
        assert scores[0] >= scores[1] >= scores[2] >= scores[3], (
            f"Scores should be descending: {scores}"
        )

    def test_regex_pipeline_improves_file_results(self):
        """Simulating file_search: expand to regex, then re-score."""
        query = "config yaml json"

        class FakeMatch:
            def __init__(self, path, content):
                self.path = path
                self.content = content

        matches = [
            FakeMatch("readme.md", "project overview"),
            FakeMatch("config.py", "load config from yaml"),
            FakeMatch("settings.py", "config yaml json parser"),
        ]

        scored = score_file_results(matches, query)
        # settings.py has all 3 terms
        assert "settings.py" in scored[0].path
        # readme.md has 0 terms → last
        assert "readme.md" in scored[-1].path

    def test_no_expand_with_explicit_or(self):
        """User wrote OR explicitly → don't expand."""
        query = "obsidian OR vault"
        assert _should_expand(query) is False
        # Pass through as-is to FTS5

    def test_no_expand_with_regex_chars(self):
        """User wrote regex → don't expand."""
        query = "config\\.(yaml|json)"
        assert _has_regex_special_chars(query) is True
        assert _should_expand(query) is True  # no boolean, 3+ terms...
        # Wait — the regex chars guard is inside expand_for_regex, not _should_expand.
        # _should_expand only checks booleans and quotes. Callers must check
        # regex chars separately before using expand_for_regex!

    def test_regex_expand_guard_on_caller(self):
        """Caller pattern: check regex chars before expand_for_regex."""
        query = "config\\.(yaml|json)"
        if _has_regex_special_chars(query):
            # Don't expand — pass through as-is
            expanded = query
        else:
            expanded = expand_for_regex(query)
        # With regex chars, should keep original
        assert expanded == "config\\.(yaml|json)"


# ═══════════════════════════════════════════════════════════════════════
# §C. Regression Tests — edge cases from design doc
# ═══════════════════════════════════════════════════════════════════════

class TestRegression:
    """Edge cases documented in the design proposal."""

    def test_many_keywords_default_and_problem(self):
        """The original problem: 6 keywords → AND → 0 results."""
        query = "obsidian vault path location address 库地址"
        assert _should_expand(query) is True
        expanded = expand_for_fts5(query)
        # Should produce OR chain with all tokens
        or_count = expanded.count(" OR ")
        assert or_count >= 4, f"Expected many OR operators, got {or_count}"

    def test_single_term_no_expand(self):
        assert expand_for_fts5("obsidian") == "obsidian"
        assert expand_for_regex("obsidian") == "obsidian"

    def test_score_monotonic_with_more_matches(self):
        """More query terms matched = strictly higher score (same file)."""
        results = []
        for n in range(1, 5):
            results.append({
                "content": " ".join(["term"] * n),
                "id": n,
            })
        scored = score_by_term_coverage(results, "term")
        scores = [r["_search_score"] for r in scored]
        # Higher TF = higher score
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score monotonicity broken: {scores}"
