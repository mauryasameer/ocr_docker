from __future__ import annotations

import html
from datetime import UTC, datetime

from meerax.report.builder import ReportBuilder, ReportSection

GOVERNANCE_BANNER = (
    "GOVERNANCE NOTICE: This report supports a human reviewer's judgment of OCR accuracy; it "
    "does not autonomously approve or reject extracted text for downstream use. Confidence "
    "scores reflect the underlying engine's own calibration, not a correctness guarantee."
)


def _run_info_html(engine_name: str, image_count: int, timestamp: str) -> str:
    return (
        f"<p class='meta'>Run Info &mdash; engine: {html.escape(engine_name)} | "
        f"images: {image_count} | generated: {html.escape(timestamp)}</p>"
    )


def build_report(results: dict, engine_name: str) -> ReportBuilder:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    report = ReportBuilder("OCR Benchmark Report", subtitle=GOVERNANCE_BANNER)

    report.add_section(
        ReportSection(
            title="Aggregate Metrics",
            metrics={
                "Average F1": round(results["average_f1"], 4),
                "Average CER": round(results["average_cer"], 4),
            },
        )
    )

    for entry in results["detailed_results"]:
        report.add_section(
            ReportSection(
                title=html.escape(entry["image"]),
                content=(
                    f"Gold: {html.escape(entry['gold'])}<br>"
                    f"Predicted: {html.escape(entry['pred'])}"
                ),
                metrics={"F1": round(entry["f1"], 4), "CER": round(entry["cer"], 4)},
            )
        )

    report.add_section(
        ReportSection(
            title="Run Info",
            content=_run_info_html(engine_name, len(results["detailed_results"]), timestamp),
        )
    )
    return report
