import difflib


class OCREvaluator:
    @staticmethod
    def calculate_f1_score(gold_text: str, predicted_text: str) -> float:
        """Word-level F1 score between gold and predicted text (case-insensitive)."""
        gold_words = gold_text.lower().split()
        pred_words = predicted_text.lower().split()

        if not gold_words and not pred_words:
            return 1.0
        if not gold_words or not pred_words:
            return 0.0

        matcher = difflib.SequenceMatcher(None, gold_words, pred_words)
        common = sum(block.size for block in matcher.get_matching_blocks())

        precision = common / len(pred_words)
        recall = common / len(gold_words)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_cer(gold_text: str, predicted_text: str) -> float:
        """Character Error Rate (CER), case-insensitive."""
        gold_text = gold_text.lower()
        predicted_text = predicted_text.lower()

        if not gold_text:
            return 0.0 if not predicted_text else 1.0

        matcher = difflib.SequenceMatcher(None, gold_text, predicted_text)
        distance = sum(
            max(i2 - i1, j2 - j1) if tag == "replace" else
            (j2 - j1 if tag == "insert" else i2 - i1)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
            if tag != "equal"
        )
        return distance / len(gold_text)

    def evaluate_batch(self, engine, test_cases: list[dict]) -> dict:
        """Evaluate engine on a list of {image_path, text} test cases."""
        results = []
        for case in test_cases:
            _, _, raw_data = engine.process_image(case["image_path"])
            predicted_text = " ".join(item["text"] for item in raw_data)
            gold_text = case["text"]
            results.append({
                "image": case["image_path"],
                "gold": gold_text,
                "pred": predicted_text,
                "f1": self.calculate_f1_score(gold_text, predicted_text),
                "cer": self.calculate_cer(gold_text, predicted_text),
            })

        avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0
        avg_cer = sum(r["cer"] for r in results) / len(results) if results else 0
        return {"average_f1": avg_f1, "average_cer": avg_cer, "detailed_results": results}
