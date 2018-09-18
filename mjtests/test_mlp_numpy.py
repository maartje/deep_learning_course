import unittest
from mjcode.modules import LinearModule, ReLUModule, SoftMaxModule, CrossEntropyModule
from mjcode.mlp_numpy import MLP
import numpy as np
import torch
import torch.nn as nn

class TestMLP(unittest.TestCase):

  def setUp(self):
      n_inputs = 64
      n_hidden = [128, 16]
      n_classes = 4
      self.mlp = MLP(n_inputs, n_hidden, n_classes)
      
      self.bsize = 7
      self.input_torch = torch.randn(self.bsize, n_inputs, requires_grad=True)
      self.input = self.input_torch.detach().numpy().T
      self.y = (np.eye(n_classes)[np.random.choice(n_classes, self.bsize)]).T

  def test_mlp_forward(self):
      out = self.mlp.forward(self.input)
      self.assertTrue(np.allclose(out.sum(axis=0), np.ones(self.bsize)))
  
  def test_mlp_backward(self):
      out = self.mlp.forward(self.input)
      
      ce_loss = CrossEntropyModule()
      loss = ce_loss.forward(out, self.y)
      dloss = ce_loss.backward(out, self.y)

      self.mlp.backward(dloss)
      self.assertNotEqual(self.mlp.modules[0].grads['weight'][21,45], 0)
      self.assertNotEqual(self.mlp.modules[0].grads['bias'][14,0], 0)
      
