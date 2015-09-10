import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  b1.shape = (1, b1.shape[-1])
  b2.shape = (1, b2.shape[-1])
  Y1 = X.dot(W1) + np.tile(b1, (X.shape[0], 1))
  Y1_relu = np.clip(Y1, 0, float('Inf'))
  Y2 = Y1_relu.dot(W2) + np.tile(b2, (Y1_relu.shape[0], 1))
  scores = Y2
  pass
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = 0
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################
  for data_index in range(X.shape[0]):
    # current_input = X[data_index, :]
    # current_input.shape = (1, current_input.shape[0])
    current_score = scores[data_index, :]
    score_exp = np.exp(current_score)
    labeled_score_exp = np.exp(current_score[y[data_index]])
    loss += -np.log(labeled_score_exp / np.sum(score_exp))
  loss /= X.shape[0]
  loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
  pass
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  grad_loss_to_score = 0
  grad_loss_to_b2 = 0
  grad_loss_to_W2 = 0
  grad_loss_to_Y1 = 0
  grad_loss_to_b1 = 0
  grad_loss_to_W1 = 0
  for data_index in range(X.shape[0]):
    current_score = scores[data_index, :]
    score_exp = np.exp(current_score)
    labeled_score_exp = np.exp(current_score[y[data_index]])
    labeled_indicator = np.zeros_like(score_exp)
    labeled_indicator[y[data_index]] = 1
    grad_loss_to_b2 += 1 / np.sum(score_exp) * score_exp - labeled_indicator
    # grad_loss_to_score += (1 / np.sum(score_exp) - 1 / labeled_score_exp) * score_exp
    # grad_loss_to_b2 += (1 / np.sum(score_exp) - 1 / labeled_score_exp) * score_exp
    current_Y1_relu = Y1_relu[data_index, :]
    current_Y1_relu.shape = (1, current_Y1_relu.shape[-1])
    temp = 1 / np.sum(score_exp) * score_exp - labeled_indicator
    temp.shape = (1, temp.shape[0])
    grad_loss_to_W2 += current_Y1_relu.T.dot(temp)
    larger_0_index = np.where(Y1[data_index, :] > 0)
    diag_indicator = np.zeros((current_Y1_relu.shape[1], current_Y1_relu.shape[1]))
    diag_indicator[larger_0_index, larger_0_index] = 1
    grad_loss_to_Y1 += (1 / np.sum(score_exp) * score_exp - labeled_indicator).dot(W2.T).dot(diag_indicator)
    grad_loss_to_b1 += (1 / np.sum(score_exp) * score_exp - labeled_indicator).dot(W2.T).dot(diag_indicator)
    current_input = X[data_index, :]
    current_input.shape = (1, current_input.shape[-1])
    temp = (1 / np.sum(score_exp) * score_exp - labeled_indicator).dot(W2.T).dot(diag_indicator)
    temp.shape = (temp.shape[0], 1)
    grad_loss_to_W1 += current_input.T.dot(temp.T)
  grad_loss_to_W1 /= X.shape[0]
  grad_loss_to_W1 += 0.5 * reg * 2 * W1
  grads['W1'] = grad_loss_to_W1
  grad_loss_to_W2 /= X.shape[0]
  grad_loss_to_W2 += 0.5 * reg * 2 * W2
  grads['W2'] = grad_loss_to_W2
  grad_loss_to_b1 /= X.shape[0]
  grads['b1'] = grad_loss_to_b1
  grad_loss_to_b2 /= X.shape[0]
  grads['b2'] = grad_loss_to_b2
  pass
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads

