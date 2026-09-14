# GOVERNANCE.md

## Intended Use

This framework supports a human reviewer's judgment of OCR extraction quality — it does not
autonomously approve, file, or act on extracted text. Every benchmark report is decision support
for someone deciding whether a given engine's output is accurate enough for a downstream use
case; OCR output should be human-verified before any automated filing or data-entry process
consumes it.

## Explainability Boundary

Per-detection confidence scores (surfaced in the Gradio app and in benchmark reports) are the
underlying OCR engine's own calibration, not an independent correctness guarantee — a
high-confidence detection can still be wrong, and a low-confidence one can still be right. F1 and
Character Error Rate (CER) scores in a benchmark report describe how closely predicted text
matched a specific gold-standard reference, not a general accuracy claim beyond that test set.

## Fairness

This framework processes document images for text extraction; it collects and uses no
demographic or protected-attribute data of any kind. A disparate-impact fairness analysis does
not apply to this domain — no fairness proxy is reported here because none would be meaningful.

## Audit Trail

Every generated benchmark report embeds a "Run Info" block recording the OCR engine used, the
number of images evaluated, and a timestamp, alongside per-image gold/predicted text and F1/CER
scores — so any given report is traceable to exactly what produced it.

## Regulatory Framing

If the documents processed through this framework contain personal data, GDPR is the applicable
lens for how that data is handled and retained — named here as the relevant framework, not a
compliance claim. This project has no other financial-services, healthcare, or HR-specific
regulatory scope.
