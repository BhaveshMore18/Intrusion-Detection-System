import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_extraction import FEATURE_NAMES, extract_features, _shannon_entropy  # noqa: E402


class TestShannonEntropy(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_shannon_entropy(b""), 0.0)

    def test_uniform_byte_is_zero_entropy(self):
        self.assertEqual(_shannon_entropy(b"\x00" * 1000), 0.0)

    def test_random_bytes_high_entropy(self):
        data = os.urandom(20000)
        entropy = _shannon_entropy(data)
        self.assertGreater(entropy, 7.5)  # near-max (8.0) for true random data


class TestExtractFeatures(unittest.TestCase):
    def _write_temp(self, data: bytes) -> str:
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.addCleanup(os.remove, path)
        return path

    def test_output_shape_and_order(self):
        path = self._write_temp(b"hello world, this is a plain text file\n" * 5)
        vec = extract_features(path)
        self.assertEqual(len(vec), len(FEATURE_NAMES))

    def test_text_file_has_low_entropy_and_not_pe(self):
        path = self._write_temp(b"the quick brown fox jumps over the lazy dog\n" * 50)
        vec = extract_features(path)
        features = dict(zip(FEATURE_NAMES, vec))
        self.assertLess(features["byte_entropy"], 5.0)
        self.assertEqual(features["is_pe"], 0)
        self.assertGreater(features["printable_ratio"], 0.9)

    def test_random_data_has_high_entropy(self):
        path = self._write_temp(os.urandom(30000))
        vec = extract_features(path)
        features = dict(zip(FEATURE_NAMES, vec))
        self.assertGreater(features["byte_entropy"], 7.5)

    def test_empty_file(self):
        path = self._write_temp(b"")
        vec = extract_features(path)
        features = dict(zip(FEATURE_NAMES, vec))
        self.assertEqual(features["file_size"], 0)
        self.assertEqual(features["byte_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
