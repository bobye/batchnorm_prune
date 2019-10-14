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

"""A binary to train CIFAR-10 using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

from tqdm import tqdm

import cifar10

from batchnorm_prune import Transformer
from batchnorm_prune.Graph import NeuralFlow

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('base_stride', 192,
                            """maximum number of channels across layers""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train_0',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_boolean('squeeze_model', True,
                            """Whether to port a squeezed model""")
tf.app.flags.DEFINE_integer('checkpoint_version', 0,
                            "the version of checkpoint to save")
tf.app.flags.DEFINE_float('l1_penalty', 0.005,
                          "the L1 penality for proximal gradient")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          "learning rate used for sgd")

def _get_train_dir(version):
  return FLAGS.train_dir + "_" + str(version)

def _prepare_checkpoint_and_flow(graph):
  dt = Transformer.Transformer(graph)
  dt.checkpoint_version = FLAGS.checkpoint_version
  if FLAGS.checkpoint_version > 0:
    dt.load_flow(_get_train_dir(dt.checkpoint_version-1) + '/pruned_flow_graph.pkl')
  cifar10.INITIAL_LEARNING_RATE = FLAGS.learning_rate
  return dt

def train(graph = tf.Graph()):
  dt = _prepare_checkpoint_and_flow(graph)
  
  """Train CIFAR-10 for a number of steps."""
  with graph.as_default():
    with graph.device('/cpu:0'):
      global_step = tf.train.get_or_create_global_step()

      # Get images and labels for CIFAR-10.
      images, labels = cifar10.distorted_inputs()    

    logits = cifar10.inference(dt, images, is_compressed = FLAGS.checkpoint_version > 0)
    dt.flow.cost = dict() # empty cost
        
    # Calculate loss.
    loss = cifar10.loss(dt, logits, labels)


    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      train_op, variable_averages = cifar10.train(dt, loss, global_step, l1_penalty = FLAGS.l1_penalty)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = variable_averages.variables_to_restore()

    # Get all bind-to variables (not using moving averages)
    var_gamma = []
    var_beta  = []
    var_weights = []
    var_biases  = []
    for name, v in variables_to_restore.items():
      if re.match(r'.*gamma/.*', name):
        var_gamma.append(v)
      if re.match(r'.*beta/.*', name):
        var_beta.append(v)
      if re.match(r'.*weights/.*', name):
        var_weights.append(v)
      if re.match(r'.*biases/.*', name):
        var_biases.append(v)
        
    group_lasso, sparsity= dt.regularization(var_gamma)
    reset_global_step = tf.assign(global_step, tf.zeros_like(global_step))
    prune_op = dt.prune(tf.get_collection('is prune'))
    
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def after_create_session(self, sess, coord):
        pass

      def before_run(self, run_context):
        self._step += 1
        if self._step % FLAGS.log_frequency == 0:          
          return tf.train.SessionRunArgs([loss,group_lasso,sparsity])  # Asks for loss value.
        
      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results[0]
          group_lasso= run_values.results[1]
          sparsity   = run_values.results[2]
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f, group lasso = %.2f, fake sparsity = %.2f, (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value, group_lasso, sparsity, 
                               examples_per_sec, sec_per_batch))

      def end(self, sess):
        sess.run(prune_op)      
        # save flow graph
        dt.flow.prune()
        dt.flow.print_summary()
        dt.save_flow(_get_train_dir(dt.checkpoint_version) + '/pruned_flow_graph.pkl')
      
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=_get_train_dir(dt.checkpoint_version),
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=config) as mon_sess:

      # training and sparisifying 
      while not mon_sess.should_stop():
        mon_sess.run(train_op)  

    # get compressed network
    if not FLAGS.squeeze_model:
      return tf.get_default_graph()
    else:
      dt.is_compressed = True
      dt.cached_flow = dt.flow
      dt.flow = NeuralFlow()
      
    with tf.variable_scope('compressed') as scope:
      logits_c = cifar10.inference(dt, images, is_compressed=True)
      loss = cifar10.loss(dt, logits_c, labels)

    all_var = variables_to_restore.values()
    all_trainable_var_compressed = [v for v in tf.trainable_variables() if v.op.name.startswith('compressed')]
    _ = variable_averages.apply(all_trainable_var_compressed)
    all_var_compressed = [v for v in tf.global_variables() if v.op.name.startswith('compressed')]

    #print(all_var)
    #print(all_var_compressed)
    assign_ops, assign_ops2 = Transformer.compress_to(dt, all_var, all_var_compressed)
    
    #print("Variables to restore from checkpoints: ", variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)      
    ckpt = tf.train.get_checkpoint_state(_get_train_dir(dt.checkpoint_version))

    variables_to_save = {}
    for v in all_var_compressed:
      vname = v.op.name.replace('compressed/','')
      variables_to_save[vname] = v
      
    variables_to_save[global_step.op.name] = global_step
    saver_c = tf.train.Saver(variables_to_save)      

    sess = tf.Session()
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      sess.run(dt.shadow[logits.op.name]) # rebuild betas
      sess.run(assign_ops)
      sess.run(assign_ops2)
      sess.run(reset_global_step)
    saver_c.save(sess, _get_train_dir(dt.checkpoint_version + 1) + '/model.ckpt-0')
    sess.close()
    
    g = tf.get_default_graph()
    return g

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.io.gfile.exists(FLAGS.train_dir):
    tf.io.gfile.rmtree(FLAGS.train_dir)
  tf.io.gfile.makedirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
