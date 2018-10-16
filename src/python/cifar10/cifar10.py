# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import copy
import numpy as np
from batchnorm_prune import Transformer
from batchnorm_prune.block_proximal_gradient_descent import BlockProximalGradientDescentOptimizer
from six.moves import urllib
import tensorflow as tf

import cifar10_input


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 125,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
			   """Weight decay for convolution filters.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.999      # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))



def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels



def cnn4(dt, images, is_compressed):
  images = dt.create(images, is_compressed=is_compressed)

  # conv1
  add_to_cost = float(dt.get_compressed_channel_size('conv2/Conv2D', 192) * 5 * 5) / float(dt.param['num_pixels'])
  conv1 = dt.conv2d_layer(images, [5, 5], 3, 96, 'conv1', use_slim=True, is_compressed=is_compressed, add_to_cost = add_to_cost)
  pool1 = dt.max_pool2d(conv1, [3, 2], 'pool1', is_compressed=is_compressed) # [, 16, 16, 96]
    
  # conv2
  add_to_cost = float(dt.get_compressed_channel_size('conv3/Conv2D', 192) * 3 * 3) / float(dt.param['num_pixels'])
  conv2 = dt.conv2d_layer(pool1, [5, 5], 96, 192, 'conv2', use_slim=True, is_compressed=is_compressed, add_to_cost = add_to_cost)
  pool2 = dt.max_pool2d(conv2, [3, 2], 'pool2', is_compressed=is_compressed) # [, 8, 8, 192]

  # conv3
  add_to_cost = float(dt.get_compressed_channel_size('fc1/Conv2D', 384) * 3 * 3) / float(dt.param['num_pixels'])
  conv3 = dt.conv2d_layer(pool2, [3, 3], 192, 192, 'conv3', use_slim=True, is_compressed=is_compressed, add_to_cost = add_to_cost)
  pool3 = dt.max_pool2d(conv3, [3, 2], 'pool3', is_compressed=is_compressed) # [, 4, 4, 192]

  # fc1
  add_to_cost = float(dt.get_compressed_channel_size('softmax_linear/Conv2D', NUM_CLASSES) * 1 * 1) / float(dt.param['num_pixels'])
  fc1 = dt.conv2d_layer(pool3, [4, 4], 192, 384, 'fc1',
                             use_slim=True, padding='VALID', is_compressed=is_compressed, add_to_cost = add_to_cost) #[, 1, 1, 384]

  # softmax
  softmax_linear = dt.conv2d_layer(fc1, [1, 1], 384, NUM_CLASSES, 'softmax_linear',
                                   activation = None, use_slim=True, is_last=True,
                                   is_compressed=is_compressed) # [, 1, 1, NUM_CLASSES]

  squeeze = dt.finalize(softmax_linear, is_compressed=is_compressed)
  return squeeze


def residual_block_helper(dt, input, name_prefix, channels, stride, is_compressed):
  add_to_cost = float(dt.get_compressed_channel_size(name_prefix + 'c1/Conv2D', channels) * 3 * 3) / float(dt.param['num_pixels'])
  conv0 = dt.conv2d_layer(input, [3, 3], input.shape[-1].value, channels, name_prefix + 'c0',
                             use_slim=True,
                             is_compressed=is_compressed,
                             add_to_cost = add_to_cost,
                             stride=stride,
                             weight_decay = FLAGS.weight_decay)
  
  conv1 = dt.conv2d_layer(conv0, [3, 3], channels, channels, name_prefix + 'c1',
                             use_slim=True,
                             is_compressed=is_compressed,
                             activation=None,
                             weight_decay = FLAGS.weight_decay,
                             no_gamma=True)

  if stride == 1:
    return dt.summation(input, conv1,name_prefix + 'sum')
  else:
    proj = dt.conv2d_layer(input, [stride, stride], input.shape[-1].value, channels, name_prefix + 'p',
                           use_slim=True,
                           is_compressed=is_compressed,
                           add_to_cost = add_to_cost,
                           stride=stride,
                           activation=None,
                           weight_decay = FLAGS.weight_decay,
                           no_gamma=True)
    return dt.summation(proj, conv1, name_prefix + 'sum')
    
  

def resnet20(dt, images, is_compressed):
  images = dt.create(images, is_compressed=is_compressed)
  print(images.shape)
  
  # first_conv
  add_to_cost = float(dt.get_compressed_channel_size('g0b0c0/Conv2D', 16) * 3 * 3) / float(dt.param['num_pixels'])
  first_conv = dt.conv2d_layer(images, [3, 3], 3, 16, 'first_conv',
                               use_slim=True,
                               is_compressed=is_compressed,
                               add_to_cost = add_to_cost,
                               weight_decay = FLAGS.weight_decay,
                               no_gamma=True)
  print(first_conv.shape)
  
  # group0 [, 32, 32, 16]
  g0b0sum = residual_block_helper(dt, first_conv, 'g0b0', 16, 1, is_compressed)
  g0b1sum = residual_block_helper(dt, g0b0sum,    'g0b1', 16, 1, is_compressed)
  g0b2sum = residual_block_helper(dt, g0b1sum,    'g0b2', 16, 1, is_compressed)
  print(g0b2sum.shape)
  
  # group1 [, 16, 16, 32]
  g1b0sum = residual_block_helper(dt, g0b2sum, 'g1b0', 32, 2, is_compressed)
  g1b1sum = residual_block_helper(dt, g1b0sum, 'g1b1', 32, 1, is_compressed)
  g1b2sum = residual_block_helper(dt, g1b1sum, 'g1b2', 32, 1, is_compressed)
  print(g1b2sum.shape)

  # group2 [, 8, 8, 64]
  g2b0sum = residual_block_helper(dt, g1b2sum, 'g2b0', 64, 2, is_compressed)
  g2b1sum = residual_block_helper(dt, g2b0sum, 'g2b1', 64, 1, is_compressed)
  g2b2sum = residual_block_helper(dt, g2b1sum, 'g2b2', 64, 1, is_compressed)
  print(g2b2sum.shape)

  # pool [, 1, 1, 64]
  pool = dt.avg_pool2d(g2b2sum, [8, 8], 'pool', padding='VALID', is_compressed=is_compressed)

  # [, 1, 1, 10]
  softmax_linear = dt.conv2d_layer(pool, [1, 1], 64, NUM_CLASSES, 'softmax_linear',
                                   activation = None,
                                   use_slim=True,
                                   is_last=True,
                                   is_compressed=is_compressed,
                                   weight_decay = FLAGS.weight_decay)
  
  squeeze = dt.finalize(softmax_linear, is_compressed=is_compressed)  
  return squeeze
  
def inference(dt, images, is_training=True, is_compressed=False):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """

  slim = tf.contrib.slim
  with slim.arg_scope([slim.batch_norm],
                      is_training=is_training):      
  
    #squeeze = cnn4(dt, images, is_compressed)
    squeeze = resnet20(dt, images, is_compressed)
    
  return squeeze

def loss(dt, logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  if dt.trainable:
    tf.add_to_collection('losses', cross_entropy_mean)
  else:
    tf.add_to_collection('losses_cpy', cross_entropy_mean)
    
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  if dt.trainable:
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
  else:
    return tf.add_n(tf.get_collection('losses_cpy'), name='total_loss_cpy')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

  
  
def train(dt, total_loss, global_step, l1_penalty = 0.0):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  print("Number of batches per epoch: %d" % int(num_batches_per_epoch))
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  # loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  if l1_penalty == 0.0:
    opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.MomentumOptimizer(lr, 0.9)
  else:
    opt = BlockProximalGradientDescentOptimizer(lr, l1_penalty, dt.flow)

  grads = opt.compute_gradients(total_loss, aggregation_method=2)

  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step, zero_debias=True)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op, variable_averages


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
