# scripts/run_benchmark.py

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ocr_engine import OCREngine
from core.evaluators.evaluator import OCREvaluator
from core.utils import save_json, format_benchmark_report

def main():
    print("🚀 Starting OCR Benchmark...")
    
    # Initialize components
    engine = OCREngine(lang='en')
    evaluator = OCREvaluator()
    
    # Load test cases
    tests_path = os.path.join("data", "gold_standard", "accuracy_tests.json")
    if not os.path.exists(tests_path):
        print(f"❌ Test cases not found at {tests_path}")
        return

    with open(tests_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    print(f"📂 Loaded {len(test_cases)} test cases.")
    
    # Run evaluation
    results = evaluator.evaluate_batch(engine, test_cases)
    
    # Print summary
    print("\n" + "="*40)
    print("📊 BENCHMARK SUMMARY")
    print("="*40)
    print(f"Average F1 Score: {results['average_f1']:.4f}")
    print(f"Average CER:      {results['average_cer']:.4f}")
    print("="*40)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("reports", f"benchmark_{timestamp}.json")
    save_json(results, report_path)
    print(f"✅ Detailed report saved to {report_path}")
    
    # Format and save Markdown report
    md_report = format_benchmark_report({
        "average_f1": results['average_f1'],
        "average_cer": results['average_cer']
    })
    md_path = os.path.join("reports", "latest_benchmark.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"✅ Markdown summary saved to {md_path}")

if __name__ == "__main__":
    main()
