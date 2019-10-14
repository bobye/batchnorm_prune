import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .Graph import NeuralFlow
import pickle
import numpy as np
import re

###########################################################
## private utility functions
def _assert_valid_name(x):
    n = x[-1].name.split(":")[0]
    assert x[-1].name.split(":")[0] in self.flow.nodes
    return n

def _get_name_without_scope(s):
    return s.split("/")[-1]
        
def _remove_prefix(s):
    m = re.match(r'(clone_[0-9]+/)?(compressed/)?(.*)', s)
    return m.group(3)

def _gather_along_axis(data, indices, axis=0):
    if not axis:
        return tf.gather(data, indices)
    rank = data.shape.ndims
    perm = [axis] + list(range(1, axis)) + [0] + list(range(axis + 1, rank))
    return tf.transpose(tf.gather(tf.transpose(data, perm), indices), perm)

###########################################################

class Transformer(object):
    from tensorflow.contrib.layers.python.layers import initializers
    
    def __init__(self, graph = tf.Graph()):

        # NeuralFlow object to be composed when the network for training is defined
        self.flow = NeuralFlow()

        # Cached NeuralFlow object when a compressed network was created
        self.cached_flow = NeuralFlow()

        # Parameters 
        self.param = {'num_pixels': 32*32,
                      'priority': 'memory'} # 'memory' mode only, 'speed' mode not yet tested

        # States
        self.is_compressed = False # whether the current network has been compressed.
        self.shadow = {}
        self.trainable = True;
        self.checkpoint_version = -1;


###########################################################
## basic common utilities
    def get_compressed_channel_size(self, name, upper):
        if self.is_compressed:
            length = len([int(v.split('_')[-1]) for v in self.cached_flow.nodes.keys()
                          if v.startswith(name) and v not in self.flow.zero_nodes])
        else:
            length = 0
        
        if length == 0:
            length = upper
        return length

    def get_variable(self, name, shape, use_slim=False, **kwargs):
        import tensorflow.contrib.slim as slim
        with tf.device('/cpu:0'):
            dtype = tf.float32 # tf.float16 if FLAGS.use_fp16 else tf.float32
            if kwargs.get('trainable') is None:
                kwargs['trainable'] = self.trainable
            if kwargs.get('dtype'):
                dtype = kwargs['dtype']
                del kwargs['dtype']        
            if use_slim:
                var = slim.model_variable(name, shape=shape, dtype=dtype, **kwargs)
            else:
                var = tf.get_variable(name, shape, dtype=dtype, **kwargs)
        return var
    def get_variable_with_weight_decay(self, name, shape, wd,
                                       use_slim=False, **kwargs):
        var = self.get_variable(name, shape, use_slim, **kwargs)
        if wd is not None and wd != 0.:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        else:
            return var
        
        if not self.is_compressed:
            tf.add_to_collection('losses', weight_decay)
        else:
            tf.add_to_collection('losses_cpy', weight_decay)
        return var
    def save_flow(self, filename = None):
        saved_filename = '/tmp/saved_flow_graph.pkl'
        if filename:
            saved_filename = filename
        file_output=open(saved_filename, 'wb')
        pickle.dump(self.flow, file_output)

    def load_flow(self, filename = None):
        saved_filename = '/tmp/saved_flow_graph.pkl'
        if filename:
            saved_filename = filename
        file_input=open(saved_filename, 'rb')
        self.cached_flow = pickle.load(file_input)

    def prune(self, is_prune_variables):
        collection = []
        for v in is_prune_variables:
            m = re.match(r'(.*)_isprune', v.op.name)
            varname = m.group(1)
            collection.append(tf.py_func(lambda a, b: self.flow.get_channel_pruned(a, b),
                                         [tf.constant(varname), v], tf.bool))          
        return collection

    def regularization(self, variables):
        edge_count = 0
        for v in variables:
            if self.flow.edgename_map.get(v.op.name):
                if self.flow.cost and self.flow.flow:
                    penality = self.flow.flow[self.flow.edgename_map[v.op.name]] * self.flow.cost[self.flow.edgename_map[v.op.name]]
                elif self.flow.flow:
                    penality = self.flow.flow[self.flow.edgename_map[v.op.name]]
                elif self.flow.cost:
                    penality = self.flow.cost[self.flow.edgename_map[v.op.name]]        
                else:
                    penality = 1.

                m = tf.abs(v) * penality
                tf.add_to_collection('group lasso', tf.reduce_sum(m))
                p = tf.equal(v, 0.0, name=v.op.name + "_isprune")
                tf.add_to_collection('is prune', p)
                tf.add_to_collection('zero edges', tf.reduce_sum(tf.cast(p, tf.float32)) * penality)
                edge_count = edge_count + p.shape[-1].value * penality
        print("weight edge count: " + str(edge_count))
        if tf.get_collection('zero edges'):
            return (tf.add_n(tf.get_collection('group lasso'), name='total_group_loss'),
                    tf.add_n(tf.get_collection('zero edges'), name='total_zero_edges') / edge_count)
        else:
            return tf.constant(0.0), tf.constant(-1)

###########################################################
# privte utilities         
    def _conv2d(self, input, filter, stride, flow_read_only=False, **kwargs):
        conv=tf.nn.conv2d(input, filter, stride, **kwargs)
        if flow_read_only:
            return conv
        name_in = input.op.name
        name_out= conv.op.name
        key_pair = (name_in, name_out)
#        flow.edgename_map[filter.name] = key_pair
        shape = filter.shape
        self.flow.cost[name_out] = input.shape[1].value * input.shape[2].value * shape[0].value * shape[1].value * shape[2].value * shape[3].value

        for j in range(conv.shape[-1].value):
            key = name_out + "_%d" % j
            self.flow.add_node(key)
            for i in range(input.shape[-1].value):
                if self.flow.nodes.get(name_in + "_%d" % i):
                    self.flow.add_edge(name_in + "_%d" % i, name_out + "_%d" % j)
        return conv

    def _bias_add(self, value, bias, flow_read_only=False, **kwargs):
        pre_activation = tf.nn.bias_add(value, bias, **kwargs)
        if flow_read_only:
            return pre_activation
        name_in= value.op.name
        name_out = pre_activation.op.name
        self.flow.cost[name_out] = value.shape[1].value * value.shape[2].value * value.shape[3].value

        for j in range(value.shape[-1].value):
            key= name_out + "_%d" % j
            self.flow.add_node(key)
            if self.flow.nodes.get(name_in + '_%d' % j):
                self.flow.add_edge(name_in + "_%d" % j, key)
        return pre_activation
    
    def _relu(self, features, flow_read_only=False, **kwargs):
        activation = tf.nn.relu(features, name='ReLU', **kwargs)
        if flow_read_only:
            return activation
        name_in = features.op.name
        name_out= activation.op.name
        for j in range(features.shape[-1].value):
            key= name_out + "_%d" % j
            self.flow.add_node(key)
            if self.flow.nodes.get(name_in + "_%d" % j):
                self.flow.add_edge(name_in + "_%d" % j, key)
        return activation

    def _pool(self, value, ksize, strides, padding, pool_func,
             flow_read_only=False, **kwargs):
        pool = pool_func(value, ksize, strides, padding, **kwargs)
        if flow_read_only:
            return pool

        name_in = value.op.name
        name_out= pool.op.name
        stride = strides[-1]
        ksize = ksize[-1]
        assert stride == 1 and ksize == 1
        length = value.shape[-1].value
        input_inc = 0
        output_inc= 0
        while input_inc < length:
            key=name_out + "_" + str(output_inc)
            self.flow.add_node(key)
            if self.flow.nodes.get(name_in + "_%d" % input_inc):
                self.flow.add_edge(name_in + "_%d" % input_inc, key)
            input_inc = input_inc + 1
            output_inc= output_inc+ 1
        return pool

    def _slim_batch_norm(self,
                         input,
                         flow_read_only=False,
                         add_to_flow = 0,
                         **kwargs):
        import tensorflow.contrib.slim as slim
        bn = tf.identity(slim.batch_norm(input, **kwargs), name='BN')
        if flow_read_only:
            return bn
        name_in = input.op.name
        name_out= bn.op.name
        key_pair = (name_in, name_out)
        if kwargs.get('center') == True and kwargs.get('scale') == True:
            self.flow.edgename_map[tf.get_variable_scope().name + '/BatchNorm/gamma'] = (name_in, name_out)
            if self.flow.flow.get(key_pair):
                self.flow.flow[key_pair] += float(add_to_flow)
            else:
                self.flow.flow[key_pair] =  float(add_to_flow)
            print("flow penality %.6f" % self.flow.flow[key_pair])
        for j in range(input.shape[-1].value):
            key= name_out + "_%d" % j
            self.flow.add_node(key)
            self.flow.add_edge(name_in + "_%d" % j, key)
        self.flow.cost[name_out] = input.shape[1].value * input.shape[2].value * input.shape[3].value
        return bn

###########################################################
## API layers    
    def relu_layer(self, features, **kwargs):
        activation = self._relu(features, flow_read_only=False, **kwargs)
        self.shadow[activation.op.name] = tf.nn.relu(self.shadow[features.op.name])
        return activation

    def max_pool_layer(self, value, ksize, stride, padding, flow_read_only, **kwargs):
        return self._pool(value, ksize, stride, padding, pool_func=tf.nn.max_pool, flow_read_only=flow_read_only, **kwargs)

    def avg_pool_layer(self, value, ksize, stride, padding, flow_read_only, **kwargs):
        return self._pool(value, ksize, stride, padding, pool_func=tf.nn.avg_pool, flow_read_only=flow_read_only, **kwargs)

    
    def summation(self, first, second, scope_name, flow_read_only=False):
        name_in1 = first.op.name
        name_in2 = second.op.name
        with tf.variable_scope(scope_name) as scope:
            summation = tf.add(first, second, name='Sum')
            name_out = summation.op.name
        if not flow_read_only:
            for i in range(first.shape[-1].value):
                key = name_out + "_%d" % i
                self.flow.add_node(key)
                self.flow.add_edge(name_in1 + "_%d" % i, key)
                self.flow.add_edge(name_in2 + "_%d" % i, key)
        self.shadow[summation.op.name] = self.shadow[first.op.name] + self.shadow[second.op.name]        
        return summation

    def concat(self, values, axis, flow_read_only=False, is_compressed=False):
        values = [x for x in values if x is not None]
        assert values
        assert axis == 3 or axis == -1
        concat = tf.concat(values=values, axis=axis, name='Concatenate')
        if not flow_read_only:
            name_out= concat.op.name
            i = 0
            for value in values:
                name_in = value.op.name
                for j in range(value.shape[axis]):
                    key = name_out + "_" + str(i)
                    self.flow.add_node(key)
                    self.flow.add_edge(name_in + "_" + str(j), key)
                    i = i + 1
        self.shadow[concat.op.name] = tf.concat(values=[self.shadow[x.op.name] for x in values], axis=0)
        return concat
    
    def pad(self, input, paddings, flow_read_only=False):
        pad = tf.pad(input, paddings)
        if not flow_read_only:
            name_in = input.op.name
            name_out = pad.op.name
            for i in range(input.shape[-1].value):
                key = name_out + "_%d" % i
                self.flow.add_node(key)
                self.flow.add_edge(name_in + "_%d" % i, key)
        self.shadow[pad.op.name] = self.shadow[input.op.name]
        return pad
    
    def slim_dropout(self, input, dropout_keep_prob, scope_name, flow_read_only=False, is_compressed=False):
        import tensorflow.contrib.slim as slim
        dropout = slim.dropout(input, dropout_keep_prob, scope=scope_name)
        if not flow_read_only:
            name_in = input.op.name
            name_out= dropout.op.name
            for i in range(input.shape[-1].value):
                key = name_out + "_%d" % i
                self.flow.add_node(key)
                self.flow.add_edge(name_in + "_%d" %i, key)
        self.shadow[dropout.op.name] = self.shadow[input.op.name] * dropout_keep_prob
        return dropout

    
    def create(self, input, is_compressed=False):
        """
        Initiate flow nodes (without adding edges) along the last dimension of tf.Tensor
        input: tf.Tensor
    """
        if hasattr(self.flow, 'name_register') and input.op.name != self.flow.name_register:
            input = tf.identity(input, name=_get_name_without_scope(self.flow.name_register))
        name = input.op.name
        for i in range(input.shape[-1].value):
            key = name + "_%d" % i
            self.flow.add_input_node(key)
        self.flow.name_register = name # register name for later name references
        self.shadow[input.op.name] = tf.constant(0.0, shape=[input.shape[-1].value])
        return input

    def finalize(self, input, is_compressed=False, no_squeeze=False):
        """
        Set terminal nodes
        input: tf.Tensor or list(tf.Tensor)
        """    
        name = input.op.name
        for i in range(input.shape[-1].value):
            self.flow.add_terminal(name + "_%d" % i)
        if no_squeeze:
            return input
    
        if input.shape.ndims == 4:
            squeeze_input = tf.squeeze(input, [1, 2], name='Final')
            self.shadow[squeeze_input.op.name] = self.shadow[input.op.name]
            return squeeze_input
        elif input.shape.ndims == 2:
            return input

    def max_pool2d(self,
                   input,
                   param_size,
                   scope_name,
                   padding='SAME',
                   is_compressed=False):
        with tf.variable_scope(scope_name) as scope:            
            pool = self.max_pool_layer(
                input, [1, param_size[0], param_size[0], 1],
                [1, param_size[1], param_size[1], 1],
                padding, False, name='MaxPool')
            self.shadow[pool.op.name] = self.shadow[input.op.name]
        return pool

    def avg_pool2d(self,
                   input,
                   param_size,
                   scope_name,
                   padding='SAME',
                   is_compressed=False):
        with tf.variable_scope(scope_name) as scope:
            pool = self.avg_pool_layer(
                input, [1, param_size[0], param_size[0], 1],
                [1, param_size[1], param_size[1], 1],
                padding, False, name='AvgPool')
            self.shadow[pool.op.name] = self.shadow[input.op.name]
        return pool
        
    def conv2d_layer(self,
                     input,
                     kernel_size,
                     channel_in,
                     channel_out,
                     scope_name,
                     activation="relu", # not support relu
                     is_last=False,
                     weight_decay=0.0,
                     stride=None,
                     dilation=None, # not supported
                     padding='SAME',
                     initializer=initializers.xavier_initializer(),
                     use_slim=True,
                     use_batch_norm=True,
                     is_compressed=False,
                     l1_penalty=None,
                     add_to_cost=0.0,
                     no_gamma_beta=False,
                     no_gamma=False):
        """
        High level wrapper for convolution layer
        """
        if input is None:
            return None

        with tf.variable_scope(scope_name) as scope:
            print(scope.name)
            if not is_compressed:
                kernel = self.get_variable_with_weight_decay('weights',
                                                             shape=[kernel_size[0], kernel_size[1], channel_in, channel_out],
                                                             wd=weight_decay,
                                                             use_slim=use_slim,
                                                             initializer=initializer)
            else:
                ind_in = sorted([int(v.split('_')[-1]) for v in self.cached_flow.nodes.keys()
                                 if (v.startswith(_remove_prefix(input.op.name))
                                     or v.startswith(_get_name_without_scope(input.op.name))
                                     or v.startswith(input.op.name.replace('compressed/InceptionV1', 'clone_0'))
                                     or v.startswith(input.op.name.replace('batch', 'fifo_queue_Dequeue'))
                                     or v.startswith(input.op.name.replace('Identity', 'dropout/mul')))
                                 and v not in self.cached_flow.zero_nodes ])
                ind_out= sorted([int(v.split('_')[-1]) for v in self.cached_flow.nodes.keys()
                                 if (re.match('.*' + _remove_prefix(scope.name) + '/Conv2D_[0-9]+', v)
                                     or re.match('.*' + _remove_prefix(scope.name) + '/convolution/BatchToSpaceND_[0-9]+', v))
                                 and v not in self.cached_flow.zero_nodes])
                assert(ind_in)
                print('output dim: {} => {}'.format(channel_out, len(ind_out)))
                if not ind_out:
                    return None

                kernel = self.get_variable_with_weight_decay('weights',
                                                             shape=[kernel_size[0], kernel_size[1],
                                                                    len(ind_in), len(ind_out)],
                                                             wd=weight_decay,
                                                             use_slim=use_slim,
                                                             initializer=initializer)
                self.shadow[kernel.op.name] = (ind_in, ind_out)
                
            if not stride:
                conv = self._conv2d(input, kernel, [1, 1, 1, 1], flow_read_only=False, padding=padding)
            else:
                conv = self._conv2d(input, kernel, [1, stride, stride, 1], flow_read_only=False, padding=padding)

            if use_slim and use_batch_norm:
                if l1_penalty is None:
                    if self.param['priority'] == 'memory':
                        cost = float(conv.shape[1].value * conv.shape[2].value +
                                     kernel_size[0] * kernel_size[1] * input.shape[3].value) / float(self.param['num_pixels'])
                    elif self.param['priority'] == 'speed':
                        cost = float(input.shape[1].value * input.shape[2].value *
                                     kernel_size[0] * kernel_size[1] * input.shape[3].value) / float(self.param['num_pixels'] * param['num_pixels'])
                else:
                    cost = l1_penalty
                if is_last:
                    cost = 0.0
                cost += add_to_cost
            
                if no_gamma_beta:
                    pre_activation = self._slim_batch_norm(
                        conv,
                        flow_read_only = False,
                        add_to_flow = cost,
                        center=False,
                        scale=False,
                        zero_debias_moving_mean=False)
                elif no_gamma:
                    pre_activation = self._slim_batch_norm(
                        conv,
                        flow_read_only = False,
                        add_to_flow = cost,
                        center=True,
                        scale=False,
                        zero_debias_moving_mean=False)
                else:
                    pre_activation = self._slim_batch_norm(
                        conv,
                        flow_read_only=False,
                        add_to_flow = cost,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=False)
                if is_compressed:
                    self.shadow[scope.name + '/BatchNorm/gamma'] = ind_out
                    self.shadow[scope.name + '/BatchNorm/beta'] = ind_out
        
            else:
                if not is_compressed:
                    biases = self.get_variable('biases',
                                               [channel_out],
                                               use_slim=use_slim,
                                               initializer = tf.constant_initializer(0.0))
                else:
                    biases = self.get_variable('biases',
                                               [len(ind_out)],
                                               use_slim=use_slim,
                                               initializer = tf.constant_initializer(0.0))
                shadow[biases.op.name] = ind_out
                pre_activation = self._bias_add(conv, biases, flow_read_only=False)

            if activation is None:
                conv = pre_activation
            else:
                conv = self._relu(pre_activation, flow_read_only=False)

        with tf.variable_scope(scope_name, reuse=True) as scope:
            if use_slim and not use_batch_norm:
                with tf.control_dependencies([self.shadow[input.op.name]]):
                    beta = biases
                    reduced_kernel = tf.reduce_sum(kernel, [0, 1])
                    beta_add = tf.squeeze(tf.matmul(tf.expand_dims(self.shadow[input.op.name],0), reduced_kernel), 0)
                    assign_ops = [tf.assign_add(beta, beta_add, use_locking=True)]                    
            if use_slim and use_batch_norm:
                moving_mean = tf.get_variable('BatchNorm/moving_mean')
                with tf.control_dependencies([self.shadow[input.op.name]]):
                    if no_gamma_beta:
                        reduced_kernel = tf.reduce_sum(kernel, [0, 1])
                        beta_add = tf.squeeze(tf.matmul(tf.expand_dims(self.shadow[input.op.name],0), reduced_kernel), 0)
                        beta = beta_add
                        assign_ops = [tf.assign_add(moving_mean, -beta_add, use_locking=True)]
                    elif no_gamma:
                        beta = tf.get_variable('BatchNorm/beta')
                        reduced_kernel = tf.reduce_sum(kernel, [0, 1])
                        beta_add = tf.squeeze(tf.matmul(tf.expand_dims(self.shadow[input.op.name],0), reduced_kernel), 0)
                        assign_ops = [tf.assign_add(moving_mean, -beta_add, use_locking=True)]
                    else:
                        beta = tf.get_variable('BatchNorm/beta')
                        gamma = tf.get_variable('BatchNorm/gamma')
                        reduced_kernel = tf.reduce_sum(kernel, [0, 1])
                        beta_add = tf.squeeze(tf.matmul(tf.expand_dims(self.shadow[input.op.name],0), reduced_kernel), 0)
                        assign_ops = [tf.assign_add(moving_mean, -beta_add, use_locking=True)]
                    
            with tf.control_dependencies(assign_ops):
                if activation == "relu":
                    beta_act = tf.nn.relu(beta)
                elif activation is None:
                    beta_act = beta
                    
            if (no_gamma_beta or no_gamma) or not use_batch_norm:
                self.shadow[conv.op.name] = tf.zeros_like(beta_act)
            else:
                self.shadow[conv.op.name] = beta_act * tf.cast(tf.equal(gamma, 0.0), tf.float32)                        
        return conv


###########################################################
## utilities for post processing before saving checkpoints

def compress_to(dt, all_var, all_var_comppressed):
    assign_ops=[]
    averages = []
    for t in all_var_comppressed:
        if t.op.name.endswith('/ExponentialMovingAverage'):
            print(t.op.name)
            s = next((x for x in all_var_comppressed
                      if t.op.name.startswith(x.op.name) and x is not t), None)
            assert s
            averages.append((s, t))
        elif re.match(r'.*weights', t.op.name):
            print(t.op.name)
            s = next((x for x in all_var
                      if x.op.name.startswith(t.op.name.replace('compressed/', ''))), None)
            assert s
            ind_in, ind_out = dt.shadow[t.op.name]
            assign_ops.append(tf.assign(t, _gather_along_axis(_gather_along_axis(s, ind_in, 2), ind_out, 3), use_locking=True))
        elif re.match(r'.*biases.*', t.op.name) and t.shape.ndims > 0:
            print(t.op.name)
            s = next((x for x in all_var
                      if x.op.name.startswith(t.op.name.replace('compressed/',''))), None)
            assert s
            ind = dt.shadow[t.op.name]
            assign_ops.append(tf.assign(t, tf.gather(s, ind), use_locking=True))
        elif re.match(r'(.*/BatchNorm)/.*', t.op.name) and t.shape.ndims > 0:
            print(t.op.name)
            matched = re.match(r'(.*?/BatchNorm)/.*', t.op.name)
            gamma_name = matched.group(1) + '/gamma'
            s = next((x for x in all_var
                      if x.op.name.startswith(t.op.name.replace('compressed/',''))), None)
            assert s
            ind = dt.shadow[gamma_name]
            assign_ops.append(tf.assign(t, tf.gather(s, ind), use_locking=True))
        elif t.shape.ndims==0:
            print(t.op.name)
            s = next((x for x in all_var
                      if x.op.name.startswith(t.op.name.replace('compressed/',''))), None)
            assert s
            assign_ops.append(tf.assign(t, s, use_locking=True))

    assign_ops_new=[]
    with tf.control_dependencies(assign_ops):
        for s, t in averages:
            assign_ops_new.append(tf.assign(t, s, use_locking=True))
    return assign_ops, assign_ops_new
    
