# Synthetic customer-insights decision memo

On the held-out template family, topic macro F1 is 0.728 and sentiment macro F1 is 0.533. These numbers verify the pipeline on controlled synthetic language; they are not realistic deployment benchmarks. Even perfect scores on this dataset would not validate a production model. At the illustrative confidence threshold, 99.4% of feedback requires human review: this baseline is not ready for automated routing. Inspect `error_audit.csv` to diagnose errors against known synthetic labels.

Use the test-only feedback dashboard to propose a manual audit of frequent negative topics. Route low-confidence cases to reviewers; do not auto-resolve complaints. Review thresholds and probabilities require calibration on a genuinely varied, labeled corpus.

Website funnel metrics show where simulated sessions drop from product view to cart to purchase. Compare mobile versus desktop as a diagnostic and propose a usability experiment, rather than attributing differences causally to device or channel. Feedback and digital sessions are separate populations.

Next: add de-identified or appropriately licensed public language, group splits by author and near-duplicate phrase, temporal drift checks, a labeling rubric, error slices, calibrated abstention and human review. Never infer NPS or actual business improvement from sentiment classification.
