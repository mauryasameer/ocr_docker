# scripts/run_benchmark.py

import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ocr_engine import OCRFactory
from core.evaluators.evaluator import OCREvaluator
from core.utils import save_json, format_benchmark_report

def main():
    parser = argparse.ArgumentParser(description="OCR Framework Performance Auditor")
    parser.add_argument("--engine", "-e", default="paddle", 
                        choices=OCRFactory.list_available_engines(),
                        help="OCR Engine to audit (default: paddle)")
    parser.add_argument("--tests", "-t", default="data/gold_standard/accuracy_tests.json",
                        help="Path to gold-standard tests JSON")
    
    args = parser.parse_args()

    print("=" * 60)
    print(f"🚀 Starting OCR Benchmark (Engine: {args.engine})")
    print("=" * 60)
    
    try:
        # Initialize components
        engine = OCRFactory.get_engine(args.engine)
        evaluator = OCREvaluator()
        
        # Load test cases
        if not os.path.exists(args.tests):
            print(f"❌ Test cases not found at {args.tests}")
            return

        with open(args.tests, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        print(f"📂 Loaded {len(test_cases)} test cases.")
        
        # Run evaluation
        results = evaluator.evaluate_batch(engine, test_cases)
        results["engine"] = args.engine
        
        # Print summary
        print("\n" + "-"*40)
        print(f"📊 {args.engine.upper()} BENCHMARK SUMMARY")
        print("-"*40)
        print(f"Average F1 Score: {results['average_f1']:.4f}")
        print(f"Average CER:      {results['average_cer']:.4f}")
        print("-"*40)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"audit_{args.engine}_{timestamp}.json")
        save_json(results, report_path)
        print(f"✅ Detailed audit trail saved to {report_path}")
        
        # Format and save Markdown report
        md_report = format_benchmark_report({
            f"{args.engine}_average_f1": results['average_f1'],
            f"{args.engine}_average_cer": results['average_cer']
        })
        md_path = os.path.join(report_dir, f"latest_{args.engine}_audit.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"✅ Markdown summary saved to {md_path}")

    except Exception as e:
        print(f"❌  Audit Failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
