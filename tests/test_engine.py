import unittest
import numpy as np
from backend.utils.text_processor import TextProcessor
from backend.utils.metrics import cosine_similarity, calculate_evaluation_metrics
from backend.deepcheck.embedder import DeepCheckEmbedder
from backend.deepcheck.vector_store import VectorStore
from backend.deepcheck.baseline import BaselineStringMatcher
from backend.deepcheck.attention_aligner import AttentionAligner
from backend.deepcheck.cross_lingual import CrossLingualDetector
from backend.deepcheck.trainer import ModelEvaluator, FineTuningTrainer


class TestDeepCheckEngine(unittest.TestCase):
    def setUp(self):
        self.embedder = DeepCheckEmbedder()
        self.vector_store = VectorStore(dim=384)
        self.baseline = BaselineStringMatcher()
        self.cross_lingual = CrossLingualDetector()

    def test_text_processor_segmentation(self):
        text = "Deep learning architectures have revolutionized natural language processing. Transformers process text in parallel."
        sentences = TextProcessor.segment_into_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("Deep learning architectures", sentences[0])

    def test_embedder_encoding(self):
        sentences = ["Artificial intelligence is transforming science.", "Machine learning processes big data."]
        vecs = self.embedder.encode_sentences(sentences)
        self.assertEqual(vecs.shape, (2, 384))
        # Verify L2 normalization
        norm = np.linalg.norm(vecs[0])
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_vector_store_add_and_search(self):
        self.vector_store.clear()
        sentence = "Quantum computing uses qubits."
        vec = self.embedder.encode_sentences([sentence])
        
        meta = [{"doc_id": "DOC-TEST", "title": "Quantum Paper", "author": "Alice", "sentence_text": sentence}]
        self.vector_store.add_vectors(vec, meta)
        
        results = self.vector_store.search(vec[0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "DOC-TEST")
        self.assertGreaterEqual(results[0]["similarity_score"], 0.90)

    def test_baseline_string_matcher(self):
        text_a = "Deep neural networks use gradient descent."
        text_b = "Deep neural networks apply gradient descent optimization."
        res = self.baseline.evaluate_pair(text_a, text_b)
        self.assertGreater(res["tfidf_cosine_similarity"], 0.30)

    def test_attention_aligner(self):
        in_sents = ["Transformers use attention mechanisms."]
        src_sents = ["Transformers deploy attention mechanisms for natural language processing."]
        in_vecs = self.embedder.encode_sentences(in_sents)
        src_vecs = self.embedder.encode_sentences(src_sents)
        
        alignments = AttentionAligner.align_sentences(in_sents, src_sents, in_vecs, src_vecs)
        self.assertEqual(len(alignments), 1)
        self.assertGreater(alignments[0]["similarity_score"], 0.10)
        self.assertTrue(len(alignments[0]["word_attentions"]) > 0)

    def test_cross_lingual_detector(self):
        res = self.cross_lingual.detect_cross_lingual_similarity("Global warming affects ecosystems.", "El calentamiento global afecta los ecosistemas.")
        self.assertIn("cross_lingual_similarity", res)

    def test_evaluation_benchmark(self):
        results = ModelEvaluator.run_evaluation_benchmark(self.embedder)
        self.assertIn("baseline", results)
        self.assertIn("deep_learning_sbert", results)

    def test_fine_tuning_trainer(self):
        res = FineTuningTrainer.train_epoch(num_epochs=3)
        self.assertEqual(len(res["history"]), 3)


if __name__ == "__main__":
    unittest.main()
