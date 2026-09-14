from src.services.report_service import GOVERNANCE_BANNER, build_report


def _sample_results():
    return {
        "average_f1": 0.85,
        "average_cer": 0.12,
        "detailed_results": [
            {"image": "data/gold_standard/test1.png", "gold": "hello world", "pred": "hello world", "f1": 1.0, "cer": 0.0},
            {"image": "data/gold_standard/test2.png", "gold": "foo bar", "pred": "foo baz", "f1": 0.5, "cer": 0.14},
        ],
    }


def test_build_report_includes_all_required_sections():
    report = build_report(_sample_results(), engine_name="paddle")
    html = report.to_html()

    assert GOVERNANCE_BANNER in html
    assert "Run Info" in html
    assert "paddle" in html
    assert "images: 2" in html
    assert "test1.png" in html
    assert "test2.png" in html
    assert "hello world" in html
    assert "0.85" in html or "0.8500" in html


def test_build_report_escapes_gold_and_predicted_text():
    results = _sample_results()
    results["detailed_results"][0]["pred"] = "<script>alert(1)</script>"
    report = build_report(results, engine_name="paddle")
    html = report.to_html()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
