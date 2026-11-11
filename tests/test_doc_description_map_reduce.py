import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.utils import (
    DOC_DESC_BATCH_MAX_TOKENS,
    DOC_DESC_SINGLE_MAX_TOKENS,
    create_clean_structure_for_description,
    flatten_structure_to_outline,
    generate_doc_description,
    is_low_signal_title,
    text_excerpt_for_outline,
    _pack_outline_batches,
)


def _title_tree(n, title_prefix="Section"):
    """Build a flat cleaned structure with n titled child nodes under one root."""
    return [{
        "title": "Root Document",
        "node_id": "0000",
        "nodes": [
            {"title": f"{title_prefix} {i:04d}", "node_id": f"{i:04d}"}
            for i in range(1, n + 1)
        ],
    }]


class TestFlattenStructure:
    def test_flattens_nested_titles_with_indent(self):
        structure = [{
            "title": "Root",
            "node_id": "0000",
            "nodes": [
                {"title": "Child", "node_id": "0001"},
            ],
        }]
        lines = flatten_structure_to_outline(structure)
        assert lines[0] == "0000: Root"
        assert lines[1] == "  0001: Child"

    def test_skips_empty_titles(self):
        structure = [{"title": "  ", "node_id": "0001"}, {"title": "Keep", "node_id": "0002"}]
        lines = flatten_structure_to_outline(structure)
        assert lines == ["0002: Keep"]

    def test_includes_text_excerpt_for_page_nodes(self):
        structure = [{
            "title": "Section / Page 2",
            "node_id": "0001",
            "text_excerpt": "Independent Technical Assessment of the railway line from Grootegeluk to Ogies",
        }]
        lines = flatten_structure_to_outline(structure)
        assert "Section / Page 2" in lines[0]
        assert "Grootegeluk" in lines[0]


class TestLowSignalTitles:
    def test_section_page_titles(self):
        assert is_low_signal_title("Section / Page 12")
        assert is_low_signal_title("## Section / Page 1")
        assert is_low_signal_title("Page 3")
        assert not is_low_signal_title("Independent Technical Assessment")

    def test_excerpt_strips_images(self):
        raw = "## Section / Page 2\n\n![img](media/x.png)\n\nSynopsis\nRCE Consultants was appointed to conduct an Independent Technical Assessment"
        excerpt = text_excerpt_for_outline(raw, max_chars=120)
        assert "media/" not in excerpt
        assert "Independent Technical Assessment" in excerpt

    def test_clean_structure_keeps_excerpt_for_page_titles(self):
        structure = [{
            "title": "Section / Page 2",
            "node_id": "0001",
            "text": "## Section / Page 2\n\n![x](a.png)\n\nRCE Consultants ITA of Grootegeluk to Ogies railway",
            "nodes": [],
        }]
        cleaned = create_clean_structure_for_description(structure)
        assert "text" not in cleaned[0]
        assert "Grootegeluk" in cleaned[0].get("text_excerpt", "")

    def test_clean_structure_skips_excerpt_for_real_titles(self):
        structure = [{
            "title": "Track Geometry Maintenance",
            "node_id": "0001",
            "text": "Long body about track geometry that should not be copied when the title is already informative.",
            "nodes": [],
        }]
        cleaned = create_clean_structure_for_description(structure)
        assert "text_excerpt" not in cleaned[0]


class TestPackBatches:
    def test_packs_by_token_budget(self):
        # Each "word" counts as roughly one token with a simple mock.
        lines = [f"line-{i} " + ("word " * 10) for i in range(10)]

        def fake_count(text, model=None):
            return len(text.split())

        with patch("pageindex.utils.count_tokens", side_effect=fake_count):
            batches = _pack_outline_batches(lines, model="test", max_tokens=40)

        assert len(batches) > 1
        assert sum(len(b) for b in batches) == len(lines)


class TestGenerateDocDescription:
    def test_small_outline_single_call(self):
        structure = _title_tree(3)
        calls = []

        def fake_count(text, model=None):
            return 10

        def fake_llm(model, prompt):
            calls.append(prompt)
            return "Short document summary."

        with patch("pageindex.utils.count_tokens", side_effect=fake_count), \
             patch("pageindex.utils.llm_completion", side_effect=fake_llm):
            result = generate_doc_description(structure, model="test")

        assert result == "Short document summary."
        assert len(calls) == 1
        assert "Document Structure:" in calls[0]

    def test_large_outline_map_then_reduce(self):
        structure = _title_tree(20)
        calls = []

        def fake_count(text, model=None):
            # Each outline line ~1000 tokens so 20 lines >> single and batch budgets.
            # Joined partials stay small.
            if "Partial summary" in text or text.startswith("Partial"):
                return 50
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if len(lines) <= 1 and "Section" not in text:
                return 50
            return max(1, len(lines)) * 1000

        def fake_llm(model, prompt):
            calls.append(prompt)
            if "Partial Document Outline:" in prompt:
                return f"Partial summary {len(calls)}"
            if "Partial Summaries:" in prompt:
                return "Final combined document summary."
            return "Unexpected"

        with patch("pageindex.utils.count_tokens", side_effect=fake_count), \
             patch("pageindex.utils.llm_completion", side_effect=fake_llm), \
             patch("pageindex.utils.DOC_DESC_SINGLE_MAX_TOKENS", 5000), \
             patch("pageindex.utils.DOC_DESC_BATCH_MAX_TOKENS", 3000):
            result = generate_doc_description(structure, model="test")

        assert result == "Final combined document summary."
        map_calls = [c for c in calls if "Partial Document Outline:" in c]
        reduce_calls = [c for c in calls if "Partial Summaries:" in c]
        assert len(map_calls) > 1
        assert len(reduce_calls) == 1

    def test_nested_reduce_when_partials_still_large(self):
        structure = _title_tree(12)
        calls = []
        stage = {"n": 0}

        def fake_count(text, model=None):
            # First-level outline is huge. Partials joined are still huge until
            # second-tier partials (shorter strings) fit the single budget.
            if "tier2" in text:
                return 100
            if "Partial summary" in text:
                # Several first-tier partials joined still over budget
                parts = [p for p in text.splitlines() if p.startswith("Partial summary")]
                if len(parts) >= 2:
                    return 9000
                return 2000
            lines = [ln for ln in text.splitlines() if "Section" in ln or "Root" in ln]
            if lines:
                return len(lines) * 2000
            return 100

        def fake_llm(model, prompt):
            calls.append(prompt)
            stage["n"] += 1
            if "Partial Document Outline:" in prompt:
                # First wave of maps vs second wave after re-batching partials
                if stage["n"] <= 6:
                    return f"Partial summary {stage['n']}"
                return f"tier2 partial {stage['n']}"
            if "Partial Summaries:" in prompt:
                return "Deep reduce final summary."
            return "Unexpected"

        with patch("pageindex.utils.count_tokens", side_effect=fake_count), \
             patch("pageindex.utils.llm_completion", side_effect=fake_llm), \
             patch("pageindex.utils.DOC_DESC_SINGLE_MAX_TOKENS", 5000), \
             patch("pageindex.utils.DOC_DESC_BATCH_MAX_TOKENS", 3000):
            result = generate_doc_description(structure, model="test")

        assert result == "Deep reduce final summary."
        assert any("Partial Summaries:" in c for c in calls)
        assert sum(1 for c in calls if "Partial Document Outline:" in c) > 2

    def test_empty_map_aborts_without_reduce(self):
        structure = _title_tree(10)
        calls = []

        def fake_count(text, model=None):
            lines = [ln for ln in text.splitlines() if ln.strip()]
            return max(1, len(lines)) * 1000

        def fake_llm(model, prompt):
            calls.append(prompt)
            return ""

        with patch("pageindex.utils.count_tokens", side_effect=fake_count), \
             patch("pageindex.utils.llm_completion", side_effect=fake_llm), \
             patch("pageindex.utils.DOC_DESC_SINGLE_MAX_TOKENS", 5000), \
             patch("pageindex.utils.DOC_DESC_BATCH_MAX_TOKENS", 3000):
            result = generate_doc_description(structure, model="test")

        assert result == ""
        assert len(calls) == 1
        assert "Partial Document Outline:" in calls[0]
        assert not any("Partial Summaries:" in c for c in calls)

    def test_northcor_sized_outline_uses_map_reduce(self):
        """805 flat sections should take the map_reduce path under default budgets."""
        structure = _title_tree(804, title_prefix="NorthCor heading with enough words")
        calls = []

        def fake_count(text, model=None):
            # ~20 tokens per heading line; 805 lines ~16k tokens > 12k single budget.
            if "Partial summary" in text or "Final" in text:
                return 80 * max(1, text.count("Partial summary"))
            return max(1, len(text.splitlines())) * 20

        def fake_llm(model, prompt):
            calls.append(prompt)
            if "Partial Document Outline:" in prompt:
                return f"Partial summary {len(calls)}"
            return "Final NorthCor-style document summary."

        with patch("pageindex.utils.count_tokens", side_effect=fake_count), \
             patch("pageindex.utils.llm_completion", side_effect=fake_llm):
            result = generate_doc_description(structure, model="test")

        assert "NorthCor" in result or result.startswith("Final")
        assert len([c for c in calls if "Partial Document Outline:" in c]) > 1
        assert any("Partial Summaries:" in c for c in calls)
