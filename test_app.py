"""Offline logic checks with a test encoder; does not validate MiniLM quality."""
import unittest
import numpy as np
from app import (HybridMovieIR, evaluate, tokenize, precision_at_k,
                 recall_at_k, reciprocal_rank)


class TestEncoder:
    def encode(self, texts, **kwargs):
        # Equal similarities exercise deterministic ties and empty sparse lists.
        return np.tile(np.array([1.0, 0.0]), (len(texts), 1))


class IRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = HybridMovieIR(encoder=TestEncoder())

    def test_normalization(self):
        self.assertEqual(tokenize('THE cafés running 2021'), ['cafe', 'run', '2021'])

    def test_metrics(self):
        ids, relevant = ['x', 'a', 'b'], {'a', 'b'}
        self.assertAlmostEqual(precision_at_k(ids, relevant, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ids, relevant, 2), 0.5)
        self.assertAlmostEqual(reciprocal_rank(ids, relevant), 0.5)
        self.assertEqual(precision_at_k(['a'], relevant, 3), 1 / 3)
        self.assertEqual(reciprocal_rank([], relevant), 0)

    def test_exact_title(self):
        for sparse in ['bm25', 'tfidf']:
            r = self.engine.search('Beyond the Clock', mode='sparse', sparse=sparse)
            self.assertEqual(r[0]['movie']['id'], 'm01')

    def test_missing_terms_and_blank(self):
        self.assertEqual(self.engine.search('   '), [])
        self.assertEqual(self.engine.search('xyzzyplugh', mode='sparse'), [])
        r = self.engine.search('xyzzyplugh')
        self.assertTrue(all(x['sparse_rrf'] == 0 for x in r))
        self.assertEqual(r[0]['movie']['id'], 'm01')

    def test_rrf_and_bounds(self):
        for r in self.engine.search('time travel', k=10):
            self.assertAlmostEqual(r['score'], r['sparse_rrf'] + r['dense_rrf'])
            self.assertGreaterEqual(r['cosine'], -1)
            self.assertLessEqual(r['cosine'], 1)
        with self.assertRaises(ValueError):
            self.engine.search('space', k=0)

    def test_evaluation(self):
        report = evaluate(self.engine)
        self.assertEqual(len(report['summary']), 3)
        self.assertEqual(len(report['per_query']), 42)
        for row in report['summary']:
            for metric in ['Precision@3', 'Recall@3', 'MRR']:
                self.assertTrue(0 <= row[metric] <= 1)


if __name__ == '__main__':
    unittest.main()
