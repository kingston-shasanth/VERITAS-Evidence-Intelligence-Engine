"""
Runs the Unified Data Pipeline to process all raw datasets,
validate schemas, filter duplicates, and export the canonical benchmark JSONL.
"""
import os
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.pipeline import UnifiedDataPipeline

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "processed")


def main():
    fever_path = os.path.join(RAW_DATA_DIR, "fever", "sample_dev.jsonl")
    averitec_path = os.path.join(RAW_DATA_DIR, "averitec", "sample_dev.json")
    multilingual_path = os.path.join(RAW_DATA_DIR, "multilingual", "diagnostic_multilingual.json")
    output_path = os.path.join(PROCESSED_DIR, "canonical_claims.jsonl")

    pipeline = UnifiedDataPipeline()

    print("[*] Ingesting and validating FEVER...")
    _, fever_rep = pipeline.load_fever(fever_path)
    print(fever_rep.summary())
    print()

    print("[*] Ingesting and validating AVeriTeC...")
    _, averitec_rep = pipeline.load_averitec(averitec_path)
    print(averitec_rep.summary())
    print()

    print("[*] Ingesting and validating Multilingual Benchmark...")
    _, multi_rep = pipeline.load_multilingual(multilingual_path)
    print(multi_rep.summary())
    print()

    global_report = pipeline.generate_global_report()
    print("[*] Global Dataset Health Summary:")
    print(global_report.summary())
    print()

    exported_count = pipeline.export_canonical_jsonl(output_path)
    print(f"[+] Successfully exported {exported_count} canonical claims to: {output_path}")


if __name__ == "__main__":
    main()
