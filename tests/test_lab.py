import pandas as pd
import unittest

from src.lab import (benchmark_demo, clean_dataset, load_sample_dataset,
                     make_model_card, parse_dataset, reproducible_split,
                     training_memory_estimate_mb)


class LabTests(unittest.TestCase):
  def test_sample_is_cleanable_and_balanced(self):
    cleaned, audit = clean_dataset(load_sample_dataset())
    self.assertGreaterEqual(len(cleaned), 40)
    self.assertEqual(cleaned.label.nunique(), 2)
    self.assertEqual(audit["clean_rows"], len(cleaned))


  def test_cleaning_removes_empty_duplicates_and_conflicts(self):
    frame = pd.DataFrame({"body": ["same", " same ", "conflict", "conflict", "   ",
                                    "a2", "a3", "a4", "b2", "b3", "b4", "a5", "a6", "a7", "b5", "b6", "b7"],
                          "category": ["a", "a", "a", "b", "b", "a", "a", "a", "b", "b", "b", "a", "a", "a", "b", "b", "b"]})
    cleaned, audit = clean_dataset(frame, "body", "category")
    self.assertEqual(len(cleaned), 13)
    self.assertNotIn("conflict", cleaned.text.tolist())
    self.assertIn("same", cleaned.text.tolist())
    self.assertEqual(audit["duplicates_removed"], 1)
    self.assertEqual(audit["conflicting_texts_removed"], 1)
    self.assertEqual(audit["empty_rows_removed"], 1)


  def test_split_is_reproducible_disjoint_and_stratified(self):
    frame, _ = clean_dataset(load_sample_dataset())
    first = reproducible_split(frame, seed=17)
    self.assertEqual(first, reproducible_split(frame, seed=17))
    self.assertTrue(set(first.train_indices).isdisjoint(first.validation_indices))
    self.assertTrue(set(first.train_indices).isdisjoint(first.test_indices))
    self.assertTrue(set(first.validation_indices).isdisjoint(first.test_indices))
    self.assertEqual(set(first.train_labels), set(first.validation_labels))
    self.assertEqual(set(first.train_labels), set(first.test_labels))


  def test_demo_is_deterministic_and_marks_simulation(self):
    frame, _ = clean_dataset(load_sample_dataset())
    a1, b1, errors1 = benchmark_demo(frame, 42)
    a2, b2, errors2 = benchmark_demo(frame, 42)
    self.assertEqual((a1, b1, errors1), (a2, b2, errors2))
    self.assertTrue(0 <= b1["macro_f1"] <= 1)
    self.assertEqual(b1["inference_ms_per_sample"], 4.9)


  def test_parse_csv_and_reject_unsupported(self):
    parsed = parse_dataset(b"text,label\nhello,yes\n", "rows.csv")
    self.assertEqual(list(parsed.columns), ["text", "label"])
    with self.assertRaisesRegex(ValueError, "CSV"):
        parse_dataset(b"x", "rows.txt")


  def test_card_is_honest_about_training_scope(self):
    frame, audit = clean_dataset(load_sample_dataset())
    baseline, _, _ = benchmark_demo(frame)
    card = make_model_card(frame, audit, baseline, 42)
    self.assertIn("**not** training a foundation model from scratch", card)
    self.assertIn("held-out test split", card)
    demo_card = make_model_card(frame, audit, baseline, 42, simulated=True)
    self.assertIn("no model was trained", demo_card)
    self.assertIn("simulated demo value", demo_card)


  def test_resource_estimate_is_bounded(self):
    self.assertTrue(64 <= training_memory_estimate_mb(1000) <= 1024)
    self.assertGreaterEqual(training_memory_estimate_mb(0), 64)


if __name__ == "__main__":
  unittest.main()

