import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import DeepNeuralNetwork  # noqa: E402


class TestDeepNeuralNetwork(unittest.TestCase):
    def test_forward_shapes(self):
        model = DeepNeuralNetwork(input_dim=5, hidden_layers=(8, 4))
        X = np.random.default_rng(0).normal(size=(10, 5))
        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (10,))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))

    def test_learns_a_linearly_separable_problem(self):
        rng = np.random.default_rng(1)
        X_pos = rng.normal(loc=3, scale=0.5, size=(150, 4))
        X_neg = rng.normal(loc=-3, scale=0.5, size=(150, 4))
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(150), np.zeros(150)])

        model = DeepNeuralNetwork(input_dim=4, hidden_layers=(8,))
        model.fit(X, y, epochs=25, batch_size=32, lr=0.05, verbose=False)

        preds = model.predict(X)
        accuracy = (preds == y).mean()
        self.assertGreater(accuracy, 0.95)

    def test_save_and_load_roundtrip(self):
        model = DeepNeuralNetwork(input_dim=3, hidden_layers=(6, 3))
        X = np.random.default_rng(0).normal(size=(5, 3))
        before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.npz")
            model.save(path)
            loaded = DeepNeuralNetwork.load(path)
            after = loaded.predict_proba(X)

        np.testing.assert_allclose(before, after, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
