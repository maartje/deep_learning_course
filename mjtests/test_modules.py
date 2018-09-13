import unittest
from mjcode.modules import LinearModule, ReLUModule
import numpy as np
import torch
import torch.nn as nn


class TestLinear(unittest.TestCase):

  def setUp(self):
    self.lin_layer_torch = nn.Linear(2,3)

    self.lin_layer = LinearModule(2,3)
    self.lin_layer.params['weight'] = self.lin_layer_torch.weight.detach().numpy()
    self.lin_layer.params['bias'] = self.lin_layer_torch.bias.detach().numpy().reshape(-1,1)

    self.x_torch = torch.randn(5,2, requires_grad=True)
    self.x = self.x_torch.detach().numpy().T


  def test_linear_init(self):
    lin_layer = LinearModule(2,3)
    self.assertTrue(np.mean(lin_layer.params['weight']) < 0.001 )
    self.assertTrue(np.mean(lin_layer.params['weight']) > - 0.001 )
    self.assertEqual(np.mean(lin_layer.params['bias']), 0)
    self.assertEqual(np.mean(lin_layer.grads['weight']), 0)
    self.assertEqual(np.mean(lin_layer.grads['bias']), 0)

  def test_linear_forward(self):
    y_torch = self.lin_layer_torch(self.x_torch).detach().numpy().T
    y = self.lin_layer.forward(self.x)
    self.assertTrue(np.array_equal(y_torch, y))

  def test_linear_backward(self):
    y_torch = self.lin_layer_torch(self.x_torch)

    dout = np.random.rand(3,5)
    dout_torch = torch.from_numpy(dout.T).to(dtype = torch.float32)
    z = (y_torch*dout_torch).sum()
    z.backward()

    wx_torch = self.lin_layer_torch.weight.grad
    bx_torch = self.lin_layer_torch.bias.grad.numpy().reshape(-1, 1)
    dx_torch = self.x_torch.grad.numpy().T

    self.lin_layer.forward(self.x)
    dx = self.lin_layer.backward(dout)

    self.assertTrue(np.allclose(dx_torch, dx))
    self.assertTrue(np.allclose(wx_torch, self.lin_layer.grads['weight']))
    self.assertTrue(np.allclose(bx_torch, self.lin_layer.grads['bias']))
    self.assertEqual(
      self.lin_layer.grads['weight'].shape, 
      self.lin_layer.params['weight'].shape
    )
    self.assertEqual(
      self.lin_layer.grads['bias'].shape, 
      self.lin_layer.params['bias'].shape
    )    

class TestRelu(unittest.TestCase):

  def setUp(self):
    self.relu_torch = nn.ReLU()

    self.relu = ReLUModule()

    self.x_torch = torch.randn(5,3, requires_grad=True)
    self.x = self.x_torch.detach().numpy().T

  def test_relu_forward(self):
    y_torch = self.relu_torch(self.x_torch).detach().numpy().T
    y = self.relu.forward(self.x)

    self.assertTrue(np.allclose(y_torch, y))

  def test_relu_backward(self):
    y_torch = self.relu_torch(self.x_torch)

    dout = np.random.rand(3,5)
    dout_torch = torch.from_numpy(dout.T).to(dtype = torch.float32)

    z = (y_torch*dout_torch).sum()
    z.backward()

    dx_torch = self.x_torch.grad.numpy().T

    self.relu.forward(self.x)
    dx = self.relu.backward(dout)

    self.assertTrue(np.allclose(dx_torch, dx))


