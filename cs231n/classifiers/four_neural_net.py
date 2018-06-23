from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import math

def batchNormal(X_input):
    X_input -= np.mean(X_input, axis=0)
    X_input /= (np.std(X_input, axis=0) + 1e-6)
    return X_input

class FourLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    if True:
        self.params['W1'] = std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['D1'] = std * np.random.randn(hidden_size1, hidden_size2)
        self.params['e1'] = np.zeros(hidden_size2)
        self.params['D2'] = std * np.random.randn(hidden_size2, hidden_size3)
        self.params['e2'] = np.zeros(hidden_size3)
        self.params['W2'] = std * np.random.randn(hidden_size3, output_size)
        self.params['b2'] = np.zeros(output_size)
    if False:
        self.params['W1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0/input_size)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['D1'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0/hidden_size1)
        self.params['e1'] = np.zeros(hidden_size2)
        self.params['D2'] = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2.0/hidden_size2)
        self.params['e2'] = np.zeros(hidden_size3)
        self.params['W2'] = np.random.randn(hidden_size3, output_size)/np.sqrt(hidden_size3)
        self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, dropout=0.5):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    D1, e1 = self.params['D1'], self.params['e1']
    D2, e2 = self.params['D2'], self.params['e2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    out1 = X.dot(W1) + b1
    relu_tmp1 = np.maximum(0.01*out1, out1)
#     U1 = (np.random.rand(*relu_tmp1.shape) < dropout)# / dropout # dropout: first dropout mask. Notice /p!
#     relu_tmp1 *= U1 # dropout: drop!
    relu_tmp1 = batchNormal(relu_tmp1) # rm batch normal
    scores_tmp1 = relu_tmp1.dot(D1) + e1
    relu_tmp2 = np.maximum(0.01*scores_tmp1, scores_tmp1)
    # relu_tmp2 = batchNormal(relu_tmp2) # rm batch normal
    # dropout:second dropout mask. Notice /p!
    U2 = (np.random.rand(*relu_tmp2.shape) < dropout)/dropout 
    relu_tmp2 *= U2 # dropout: drop!
    relu_tmp2 = batchNormal(relu_tmp2)
    scores_tmp2 = relu_tmp2.dot(D2) + e2
    relu_tmp3 = np.maximum(0.01*scores_tmp2, scores_tmp2)
    relu_tmp3 = batchNormal(relu_tmp3) # rm batch normal
    scores = relu_tmp3.dot(W2) + b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # loss1.0
    # correct_class_score = scores[np.arange(N), y].reshape(N,1)
    # exp_sum = np.sum(np.exp(scores), axis = 1).reshape(N, 1)
    # loss = np.sum(np.log(exp_sum) - correct_class_score)
    #############################################################################
    # loss1.1
#     f = scores - np.max(scores, axis = 1, keepdims = True)
#     loss = -f[range(N), y].sum() + np.log(np.exp(f).sum(axis = 1)).sum()
    #############################################################################
    # loss1.2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(N),y])
    loss = np.sum(correct_logprobs)
    #############################################################################
    # loss + reg
    loss = loss / N
    # regression1
    loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) +  0.5 * reg * np.sum(D1 * D1)
    # regression2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # out layer
    # loss1.0
    # p2 = np.exp(scores) / exp_sum
    #############################################################################
    # loss1.1
    # p2 = np.exp(f) / np.exp(f).sum(axis = 1, keepdims = True)
    #############################################################################
    # loss1.2
    p2 = probs
    #############################################################################
    p2[np.arange(N), y] += -1
    p2 /= N  #(N, C)
#     print('p2:', p2)
    # hiddlen3 layer
    dW2 = relu_tmp3.T.dot(p2)
    dW2 += reg * W2
    grads['W2'] = dW2
    grads['b2'] = np.sum(p2, axis = 0)

    # hiddlen2 layer
    p2_tmp = p2.dot(W2.T)
    p2_tmp = (relu_tmp3 > 0.01*relu_tmp3) * p2_tmp
#     p2_tmp[relu_tmp3 <= 0.00001] = 0
    dD2 = relu_tmp2.T.dot(p2_tmp)
    dD2 += reg * D2
    grads['D2'] = dD2
    grads['e2'] = np.sum(p2_tmp, axis = 0)

    # hiddlen1layer
    p1_tmp = p2_tmp.dot(D2.T)
#     p1_tmp[relu_tmp2 <= 0.00001] = 0
    p1_tmp = (relu_tmp2 > 0.01*relu_tmp2) * p1_tmp
    dD1 = relu_tmp1.T.dot(p1_tmp)
    dD1 += reg * D1
    grads['D1'] = dD1
    grads['e1'] = np.sum(p1_tmp, axis = 0)

    p1 = p1_tmp.dot(D1.T)
#     p1[relu_tmp1 <= 0.00001] = 0
    p1 = (relu_tmp1 > 0.01*relu_tmp1) * p1
    dW1 = X.T.dot(p1)
    dW1 += reg * W1
    grads['W1'] = dW1
    grads['b1'] = np.sum(p1, axis = 0)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-1, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    #############################################################################
    self.params['learning_rate'] = learning_rate
    self.params['learning_rate_decay'] = learning_rate_decay
    self.params['reg'] = reg
    self.params['batch_size'] = batch_size
    #############################################################################
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    loss = 100000
    old_loss = None
    print('iterations_per_epoch:{} num_train:{} batch_size:{} num_iters:{}'
          .format(iterations_per_epoch, num_train, batch_size, num_iters))
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      # 有放回的取样
      mask = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[mask]
      y_batch = y[mask]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      old_loss = loss
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      if loss - old_loss > 0.4 :
        print('loss ascend ++ old_loss：{}  loss:{}'.format(old_loss, loss))
        break 
      if len(loss_history) > 101 and abs(loss - loss_history[-100]) < 0.1:
        print('loss descent too slow 100 loss ago：{}  loss:{}'.format(loss_history[-100], loss))
        break
      loss_history.append(loss)

      if loss < 1.0:
        print('loss is enough, stop and validation')
        break

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      self.params['D1'] -= learning_rate * grads['D1']
      self.params['e1'] -= learning_rate * grads['e1']
      self.params['D2'] -= learning_rate * grads['D2']
      self.params['e2'] -= learning_rate * grads['e2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################
      if math.isnan(loss) or loss < 1e-5:
            print('loss miss')
            break
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

       ######################################################################
       #               每个epoch令学习率衰减
       ######################################################################
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    best_y_pred = 0.0
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    out1_pred = X.dot(self.params['W1']) + self.params['b1']
    out1_pred = np.maximum(0.1*out1_pred, out1_pred)
    out2_pred = out1_pred.dot(self.params['D1']) + self.params['e1']
    out2_pred = np.maximum(0.1*out2_pred, out2_pred)
    out3_pred = out2_pred.dot(self.params['D2']) + self.params['e2']
    out3_pred = np.maximum(0.1*out3_pred, out3_pred)
    scores_pred = out3_pred.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores_pred, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    if False:
        nNote = ['W1', 'D1', 'D2', 'W2', 'b1', 'e1', 'e2', 'b2']
        if y_pred.any() > best_y_pred:
            best_y_pred = y_pred
            self.params['best_y_pred'] = best_y_pred
            # 从字典写入csv文件
            import csv
            csvFile3 = open('csvFile3.csv','a', newline='')
            writer2 = csv.writer(csvFile3)
            for key in self.params:
                if key not in nNote:
                    writer2.writerow([key, self.params[key]])
            csvFile3.close()
    return y_pred
