# coding: utf-8

"""
线性模型
"""

import tensorflow as tf


class Linear:
    """
    确定几层，每层神经元个数
    """
    def __init__(self, batched_input, hparams):
        self.batched_input = batched_input
        self.hparams = hparams

        if not hparams.activation:
            self.activation = tf.nn.relu  # 默认

        if not hparams.initializer:
            self.initializer = tf.initializers.variance_scaling(scale=1.,
                                                                mode='fan_in',
                                                                distribution='normal',
                                                                seed=None)  # 默认

    def _add_layer(self, num_units, inp, layer_id):
        with tf.variable_scope('layer_%d' % layer_id):
            kernel = tf.get_variable('kernel',
                                     [tf.shape(inp)[-1], num_units],
                                     initializer=self.initializer)
            bias = tf.get_variable('bias', [num_units], initializer=tf.zeros_initializer())

        out = tf.matmul(inp, kernel) + bias
        return self.activation(out)

    def _build_graph(self):
        inp = self.batched_input.x

        for i, num_units in enumerate(self.hparams.num_units):
            inp = self._add_layer(num_units, inp, i+1)

        # 最后一层softmax
        with tf.variable_scope('softmax'):
            shape = [tf.shape(inp)[-1],
                     1 if self.hparams.num_class == 2 else self.hparams.num_class]
            kernel = tf.get_variable('kernel',
                                     shape,
                                     initializer=self.initializer)
            logits = tf.




