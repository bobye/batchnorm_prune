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

"""Evaluation for CIFAR-10.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import re

import numpy as np
import tensorflow as tf

import cifar10

from batchnorm_prune import Transformer
from batchnorm_prune.Graph import NeuralFlow 
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('checkpoint_version', 1,
                            "the version of checkpoint to save")
tf.app.flags.DEFINE_integer('checkpoint_iter', -1,
                            "the iteration number of checkpoint to load")
tf.app.flags.DEFINE_boolean('is_compressed', True,
                            "the iteration number of checkpoint to load")


def eval_once(dt, saver, summary_writer, top_k_op, loss_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  
  with tf.Session(config = config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir + '_' + str(dt.checkpoint_version))
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      if FLAGS.checkpoint_iter >= 0:
        saver.restore(sess, FLAGS.checkpoint_dir + '_' + str(dt.checkpoint_version) + '/model.ckpt-' + str(FLAGS.checkpoint_iter))
      else:
        saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      loss_sum = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        loss_sum += sess.run(loss_op) * FLAGS.batch_size
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.4f' % (datetime.now(), precision))

      # Compute loss
      averaged_loss = loss_sum / total_sample_count
      print('%s: test loss = %.4f' %(datetime.now(), averaged_loss))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary.value.add(tag='Averaged Loss', simple_value=averaged_loss)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  dt = Transformer.Transformer()
  dt.checkpoint_version = FLAGS.checkpoint_version
  dt.trainable = False
  
  #with dt.graph.as_default() as g:
  with tf.Graph().as_default() as g:
    if FLAGS.is_compressed:
      dt.load_flow('/tmp/cifar10_train_' + str(dt.checkpoint_version-1) + '/pruned_flow_graph.pkl')   # prefetch flow graph
      dt.flow = dt.cached_flow
    else:
      dt.flow= NeuralFlow()

    with g.device('/cpu:0'):      
      # Get images and labels for CIFAR-10.
      eval_data = FLAGS.eval_data == 'test'
      images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.    
    logits = cifar10.inference(dt, images, is_training=False, is_compressed=FLAGS.is_compressed)

    total_params, _ = slim.model_analyzer.analyze_vars(slim.get_model_variables())
    print('total params: %d' % total_params)
    
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Calculate loss.
    loss_op = cifar10.loss(dt, logits, labels)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    variables_to_restore_simple = tf.global_variables()

    
    if FLAGS.is_compressed:
      saver = tf.train.Saver(variables_to_restore_simple)
    else:
      saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    
    while True:
      eval_once(dt, saver, summary_writer, top_k_op, loss_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)      

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
