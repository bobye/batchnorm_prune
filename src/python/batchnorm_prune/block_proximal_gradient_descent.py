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

"""GradientDescent for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, linalg_ops, control_flow_ops, state_ops, array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

class BlockProximalGradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.
  """

  def __init__(self,
               learning_rate,
               group_lasso_strength,
               flow,
               finetuning_mode=False,
               use_weight_norm=False,
               use_locking=False,
               name="BlockProximalGradientDescent"):
    """Construct a new gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(BlockProximalGradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._flow = flow
    self._group_lasso_strength = group_lasso_strength
    self._finetuning_mode = finetuning_mode
    self._use_weight_norm = use_weight_norm
    
  def _apply_dense(self, grad, var):
    lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    update_op = training_ops.apply_gradient_descent(
      var,
      lr,
      grad,
      use_locking=self._use_locking).op

    if self._flow.edgename_map.get(var.op.name):
        with ops.control_dependencies([update_op]):
            key = self._flow.edgename_map[var.op.name]
            if self._flow.flow:
              threshold = math_ops.cast(ops.convert_to_tensor(self._flow.flow[key] *
                                                              self._learning_rate_tensor *
                                                              self._group_lasso_strength_tensor),
                                        var.dtype.base_dtype)
            else:
              threshold = math_ops.cast(ops.convert_to_tensor(self._learning_rate_tensor *
                                                              self._group_lasso_strength_tensor),
                                        var.dtype.base_dtype)
              
            norm = math_ops.maximum(math_ops.abs(var), 1E-16)
            mask = math_ops.maximum(1.0 - (threshold / norm), 0.)
            new_var = math_ops.multiply(var, mask)
            shrinkage = state_ops.assign(var, new_var)
        return shrinkage
    else:
        return update_op

  def _resource_apply_dense(self, grad, handle):
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return resource_variable_ops.resource_scatter_add(
        handle.handle, indices, -grad * self._learning_rate)

  def _apply_sparse_duplicate_indices(self, grad, var):
    delta = ops.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._group_lasso_strength_tensor = ops.convert_to_tensor(self._group_lasso_strength,
                                                              name="group_lasso_strength")
