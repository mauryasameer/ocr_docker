# core/evaluators/evaluator.py

import difflib

class OCREvaluator:
    @staticmethod
    def calculate_f1_score(gold_text, predicted_text):
        """
        Calculates a simple word-level F1 score between gold and predicted text.
        Case-insensitive by default.
        """
        gold_words = gold_text.lower().split()
        pred_words = predicted_text.lower().split()
        
        if not gold_words and not pred_words:
            return 1.0
        if not gold_words or not pred_words:
            return 0.0

        common = 0
        gold_word_count = len(gold_words)
        pred_word_count = len(pred_words)
        
        matcher = difflib.SequenceMatcher(None, gold_words, pred_words)
        for block in matcher.get_matching_blocks():
            common += block.size

        precision = common / pred_word_count if pred_word_count > 0 else 0
        recall = common / gold_word_count if gold_word_count > 0 else 0
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def calculate_cer(gold_text, predicted_text):
        """
        Calculates Character Error Rate (CER).
        Case-insensitive.
        """
        gold_text = gold_text.lower()
        predicted_text = predicted_text.lower()

        if not gold_text:
            return 0.0 if not predicted_text else 1.0
            
        matcher = difflib.SequenceMatcher(None, gold_text, predicted_text)
        # Fix: distance should be sum of insertions, deletions, and substitutions
        # matcher.get_opcodes() gives (tag, i1, i2, j1, j2)
        # 'replace': max(len1, len2) changes? No, standard is substitution count.
        # For simplicity in OCR, distance = sum of edits
        distance = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                distance += max(i2 - i1, j2 - j1)
            elif tag == 'insert':
                distance += j2 - j1
            elif tag == 'delete':
                distance += i2 - i1
        return distance / len(gold_text)

    def evaluate_batch(self, engine, test_cases):
        """
        Evaluates the engine on a batch of test cases.
        Each test case: {"image_path": "...", "text": "..."}
        """
        results = []
        for case in test_cases:
            img_path = case["image_path"]
            gold_text = case["text"]
            
            _, _, raw_data = engine.process_image(img_path)
            predicted_text = " ".join([item["text"] for item in raw_data])
            
            f1 = self.calculate_f1_score(gold_text, predicted_text)
            cer = self.calculate_cer(gold_text, predicted_text)
            
            results.append({
                "image": img_path,
                "gold": gold_text,
                "pred": predicted_text,
                "f1": f1,
                "cer": cer
            })
            
        avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0
        avg_cer = sum(r["cer"] for r in results) / len(results) if results else 0
        
        return {
            "average_f1": avg_f1,
            "average_cer": avg_cer,
            "detailed_results": results
        }
