"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {
      'weight': np.random.normal(loc=0, scale=0.0001, 
                                 size=(out_features, in_features)), 
      'bias': np.zeros((out_features, 1))
    }
    self.grads = {
      'weight': np.zeros((out_features, in_features)), 
      'bias': np.zeros((out_features, 1))}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. 
    They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    out = np.matmul(self.params['weight'], x) + self.params['bias']
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x_3d = np.expand_dims(self.x, axis=2)
    dout_3d = np.expand_dims(dout.T, axis=0)
    dw = (x_3d * dout_3d).sum(axis = 1)
    self.grads['weight'] = dw.T 
    self.grads['bias'] = dout.sum(axis=1).reshape(-1,1)
    dx = np.matmul(self.params['weight'].T, dout)
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    relu = np.vectorize(lambda n: max(n, 0.))
    out = relu(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    drelu = np.vectorize(lambda n: 1. if n > 0 else 0.)
    dx = drelu(self.x) * dout

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x_exp = np.exp(x)
    out = x_exp / x_exp.sum(axis=0)
    self.y = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx_list = []
    for b_index in range(self.y.shape[1]):
      y_i = self.y[:, b_index].reshape(-1,1)
      #print(self.y, y_i)
      dx = - np.matmul(y_i, y_i.T) 
      dx[np.diag_indices_from(dx)] = (y_i*(1-y_i)).reshape(-1)
      dx = dx * (dout[:, b_index].reshape(-1,1))
      dx = np.sum(dx, axis=0).reshape(-1,1)
      dx_list.append(dx)
    dx = np.hstack(dx_list)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = - np.log(np.sum(x * y, axis=0))
    out = out.mean()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    r = y * x
    dx = - np.divide(1, r, out=np.zeros_like(r), where=r!=0) 
    dx = dx/y.shape[1]
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx
