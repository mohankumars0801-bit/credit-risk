import unittest

import numpy as np
import pandas as pd

from src.backend.ai_model import AIModel


class ThresholdSelectionTests(unittest.TestCase):
    def test_maximizes_recall_without_exceeding_false_positive_cap(self):
        target = pd.Series([0, 0, 1, 1, 1])
        probabilities = np.array([0.15, 0.25, 0.8, 0.6, 0.4])

        threshold = AIModel._select_threshold_with_fp_cap(
            target, probabilities, max_false_positives=0
        )
        predicted = (probabilities >= threshold).astype(int)

        self.assertEqual(int(((predicted == 1) & (target.to_numpy() == 0)).sum()), 0)
        self.assertEqual(int(((predicted == 0) & (target.to_numpy() == 1)).sum()), 0)


if __name__ == '__main__':
    unittest.main()