import unittest
import mjcode.train_mlp_numpy 
from mjcode.train_mlp_numpy import accuracy, FLAGS
import numpy as np

class TestTrainMLPNumpy(unittest.TestCase):

  def test_accuracy(self):
    predictions = np.array([
        [0.1, 0.3, 0.5, 0.1],
        [0.4, 0.3, 0.1, 0.2],
        [0.2, 0.4, 0.0, 0.4],
        [0.0, 0.4, 0.0, 0.6],
        [0.1, 0.7, 0.1, 0.1]
    ])
    targets = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])
    acc = accuracy(predictions, targets)
    self.assertEqual(acc, 3./5)

