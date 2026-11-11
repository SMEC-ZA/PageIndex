"""Unit tests for VLM / deep-heading outline filters in md_to_tree extraction."""
from pageindex.page_index_md import extract_nodes_from_markdown


def test_skips_vlm_noise_titles_and_keeps_real_sections():
    md = "\n".join(
        [
            "# Doc Title",
            "",
            "## 2.1 North Corridor Overview",
            "Real section body.",
            "### Map Description",
            "Caption prose that should stay under 2.1.",
            "### Type: Map",
            "More caption.",
            "## 2.2 ITA Assessment Area",
            "Next section.",
        ]
    )
    nodes, _ = extract_nodes_from_markdown(md)
    titles = [n["node_title"] for n in nodes]
    assert "Doc Title" in titles
    assert "2.1 North Corridor Overview" in titles
    assert "2.2 ITA Assessment Area" in titles
    assert "Map Description" not in titles
    assert "Type: Map" not in titles


def test_keeps_canonical_image_description_and_resume_nodes():
    md = "\n".join(
        [
            "# Doc Title",
            "",
            "## 2.1 North Corridor Overview",
            "Real section body.",
            "##### Image description",
            "Network map OCR prose with place names.",
            "##### Resume",
            "Body text after the caption.",
            "### Key Features",
            "Should still be skipped as VLM noise.",
            "## 2.2 Next Section",
            "More real content.",
        ]
    )
    nodes, _ = extract_nodes_from_markdown(md)
    titles = [n["node_title"] for n in nodes]
    assert "Image description" in titles
    assert "Resume" in titles
    assert "Key Features" not in titles
    assert "2.1 North Corridor Overview" in titles
    assert "2.2 Next Section" in titles


def test_skips_non_numbered_h5_keeps_numbered_h5():
    md = "\n".join(
        [
            "# Doc",
            "## 2. System",
            "##### Random deep crumb",
            "Noise body.",
            "##### 2.6.1 Ellisras – Thabazimbi",
            "Real deep numbered section.",
            "###### Key Features",
            "Should be skipped as VLM noise too.",
        ]
    )
    nodes, _ = extract_nodes_from_markdown(md)
    titles = [n["node_title"] for n in nodes]
    assert "2. System" in titles
    assert "Random deep crumb" not in titles
    assert "2.6.1 Ellisras – Thabazimbi" in titles
    assert "Key Features" not in titles


if __name__ == "__main__":
    test_skips_vlm_noise_titles_and_keeps_real_sections()
    test_keeps_canonical_image_description_and_resume_nodes()
    test_skips_non_numbered_h5_keeps_numbered_h5()
    print("OK")
