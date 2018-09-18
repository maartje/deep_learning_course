"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

from torch import optim
import torch
import torch.nn as nn


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  predicted_values = predictions[targets == 1]
  max_values = predictions.max(axis=1)
  prediction_results = (predicted_values == max_values)
  accuracy = np.sum(prediction_results)/len(prediction_results)
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  ce_loss = nn.CrossEntropyLoss()
  convnet = ConvNet(3,10)
  optimizer = optim.Adam(convnet.parameters(), lr = FLAGS.learning_rate)

  c10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  test_data = c10['test'].images
  test_data = torch.tensor(test_data)

  acc_values = []
  loss_values = []


  for i in range(FLAGS.max_steps): #range(FLAGS.max_steps) 
    x, y = c10['train'].next_batch(FLAGS.batch_size)
    y = y.argmax(axis=1)
    x = torch.tensor(x)
    y = torch.tensor(y)

    optimizer.zero_grad()
    out = convnet(x)
    loss = ce_loss(out, y)
    loss.backward()
    optimizer.step()  
    loss_values.append(loss.item())

    # evaluate
    if i % FLAGS.eval_freq == 0: 
      with torch.no_grad():
        predictions = convnet(test_data).detach().numpy()
        targets = c10['test'].labels
        acc = accuracy(predictions, targets)
        print('acc', acc, 'loss', loss.item())
        acc_values.append(acc)

  # save loss and accuracy to file
  with open('accuracy_cnn.txt', 'a') as f_acc:
    print (acc_values, file=f_acc)
  with open('loss_cnn.txt', 'a') as f_loss:
    print (loss_values, file=f_loss)

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()