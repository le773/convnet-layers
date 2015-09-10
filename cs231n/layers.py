# coding=utf-8
import numpy as np
import scipy.io as sio
from cs231n.im2col import *

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = np.zeros((x.shape[0], w.shape[1]))
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  batch_size = x.shape[0]
  for batch_index in range(batch_size):
    current_input = x[batch_index]
    out[batch_index, :] = np.ravel(current_input).dot(w) + b
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache

def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  get_sum_matrix = np.ones((w.shape[1], 1))
  batch_size = x.shape[0]
  for batch_index in range(batch_size):
    get_sum_matrix = dout[batch_index]
    db += get_sum_matrix
    get_sum_matrix.shape = [1, get_sum_matrix.shape[0]]
    current_input = x[batch_index]
    current_input = np.ravel(current_input)
    current_input.shape = (1, current_input.shape[0])
    dw += current_input.T.dot(get_sum_matrix)
    dx[batch_index] = np.reshape(w.dot(get_sum_matrix.T), x[batch_index].shape)
  # dw /= batch_size
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.clip(x, 0, float('Inf'))
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dout_vec = np.ravel(dout)
  dout_vec.shape = (1, dout_vec.shape[0])
  x_vec = np.ravel(x)
  diag_matrix = np.zeros((x_vec.shape[0], x_vec.shape[0]))
  diag_matrix[np.where(x_vec > 0), np.where(x_vec > 0)] = 1
  dx = dout_vec.dot(diag_matrix)
  dx = dx.reshape(x.shape)
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  file_num, _, filt_height, file_width = w.shape
  out_height = 1 + (H + 2 * pad - filt_height) / stride
  out_width = 1 + (W + 2 * pad - file_width) / stride
  x_cols =im2col_indices(x, filt_height, file_width, pad, stride)
  filt_cols = w.reshape(file_num, -1)
  b_cols = b
  b_cols.shape = (b_cols.shape[0], 1)
  out = filt_cols.dot(x_cols) + np.tile(b, x_cols.shape[1])
  out = np.reshape(out, (file_num, out_height, out_width, N))
  out = out.transpose((3, 0, 1, 2))
  cache = (x, w, b, conv_param)
  return out, cache
  # #############################################################################
  # # TODO: Implement the convolutional forward pass.                           #
  # # Hint: you can use the function np.pad for padding.                        #
  # #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  filt_num, filt_channel, filt_height, file_width = w.shape
  dout_reshpe = np.transpose(dout, (1, 2, 3, 0))
  dout_reshpe = dout_reshpe.reshape(dout_reshpe.shape[0], -1)
  filt_col = w.reshape(filt_num, -1)
  dx_col = filt_col.T.dot(dout_reshpe)
  dx = col2im_indices(dx_col, x.shape, filt_height, file_width, pad, stride)
  x_cols = im2col_indices(x, filt_height, file_width, pad, stride)
  dw_col = dout_reshpe.dot(x_cols.T)
  dw = dw_col.reshape(w.shape)
  db = np.sum(dout_reshpe, axis=1)
  db.shape = (db.shape[0], 1)
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  (N, C, H, W) = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  x_cols = im2col_indices(x, pool_height, pool_width, 0, stride)
  out_height = 1 + (H - pool_height) / stride

  x_cols_transpose = x_cols.T
  x_cols_transpose = x_cols_transpose.reshape(x_cols_transpose.shape[0], -1, pool_height * pool_width,)
  x_cols_transpose_pooled = np.max(x_cols_transpose, axis=2)
  out = x_cols_transpose_pooled.T.reshape(C, out_height, -1, N)
  out = out.transpose((3, 0, 1, 2))
  cache = (x, pool_param)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache, d_correct):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  # x, pool_param = cache
  # (N, C, H, W) = x.shape
  # #############################################################################
  # # TODO: Implement the max pooling backward pass                             #
  # #############################################################################
  # # dout = dout.transpose((1, 2, 3, 0))
  # dout = dout.reshape(-1, C)
  # dout = np.ravel(dout)
  #
  # pool_height = pool_param['pool_height']
  # pool_width = pool_param['pool_width']
  # stride = pool_param['stride']
  # x_cols = im2col_indices(x, pool_height, pool_width, 0, stride)
  #
  # x_cols_transpose = x_cols.T
  # x_shape1 = x_cols_transpose.shape
  # x_cols_transpose = x_cols_transpose.reshape(x_cols_transpose.shape[0], -1, pool_height * pool_width,)
  # x_cols_transpose = x_cols_transpose.transpose((2, 1, 0))
  # x_shape2 = x_cols_transpose.shape
  # x_cols_transpose = x_cols_transpose.reshape((x_cols_transpose.shape[0], -1)).T
  # max_index = np.argmax(x_cols_transpose, axis=1)
  # index_bias = np.arange(max_index.shape[0]) * pool_height * pool_width
  # max_index += index_bias
  # multi_matrix = np.zeros((dout.shape[0], np.prod(x_cols_transpose.shape)))
  # multi_matrix[np.arange(dout.shape[0]), index_bias] = 1
  # dout.shape = (dout.shape[0], 1)
  # dx = multi_matrix.T.dot(dout)
  # dx = dx.reshape(x_cols_transpose.shape[0], -1).T
  # dx = dx.reshape(x_shape2)
  # dx = dx.transpose((2, 1, 0))
  # dx = dx.reshape(x_shape1).T
  # dx = col2im_indices(dx, x.shape, pool_height, pool_width, 0, stride)
  # dx = None
  # dout = np.ravel(dout)
  # x, pool_param = cache
  # (N, C, H, W) = x.shape
  # pool_height = pool_param['pool_height']
  # pool_width = pool_param['pool_width']
  # stride = pool_param['stride']
  # x_cols = im2col_indices(x, pool_height, pool_width, 0, stride).T
  # x_shape1 = x_cols.shape
  # x_cols = np.reshape(x_cols, (x_cols.shape[0], C, -1))
  # max_index = np.argmax(x_cols, axis=2)
  # max_index = np.ravel(max_index)
  # max_bias = np.arange(max_index.size) * pool_height * pool_width
  # max_index += max_bias
  # # multi_matrix = np.zeros((dout.size, x.size))
  # # multi_matrix[np.arange(dout.size), max_index] = 1
  # # dx = multi_matrix.T.dot(dout)
  # dx = np.zeros((np.prod(x.shape), 1))
  # dout.shape = (dout.shape[0], 1)
  # dx[max_index] = dout
  # dx = np.reshape(dx, x_shape1).T
  # dx = col2im_indices(dx, x.shape, pool_height, pool_width, 0, stride)

  dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
  x, pool_param = cache
  (N, C, H, W) = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  x = x.reshape((N*C, 1, H, W))
  x_cols = im2col_indices(x, pool_height, pool_width, 0, stride)
  max_index = np.argmax(x_cols, axis=0)
  dx = np.zeros_like(x_cols)
  dx[max_index, np.arange(max_index.size)] = dout_reshaped
  dx = col2im_indices(dx, x.shape, pool_height, pool_width, 0, stride)
  dx = dx.reshape((N, C, H, W))
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

