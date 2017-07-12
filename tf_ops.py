'''

   Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math


'''
   Batch normalization
   https://arxiv.org/abs/1502.03167
'''
def bn(x):
   return tf.layers.batch_normalization(x)


'''
   Instance normalization
   https://arxiv.org/abs/1607.08022
'''
def instance_norm(x, epsilon=1e-5):
   mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
   return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


'''
   Transpose convolution, but resizing first then performing conv2d with stride 1
   See http://distill.pub/2016/deconv-checkerboard/
'''
def upconv2d(x, filters, kernel_size, name):

   height = x.get_shape()[1]
   width  = x.get_shape()[2]

   # resize image using method of nearest neighbor
   x_resize = tf.image.resize_nearest_neighbor(enc_conv8, [height*2, width*2])

   # conv with stride 1
   out = tf.layers.conv2d(x_resize, filters, kernel_size, strides=1, name=name)

   return out


######## activation functions ###########
'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2):
   return tf.maximum(leak*x, x)


'''
   Like concatenated relu, but with elu
   http://arxiv.org/abs/1603.05201
'''
def concat_elu(x):
   axis = len(x.get_shape())-1
   return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))

'''
   Concatenated ReLU
   http://arxiv.org/abs/1603.05201
'''
def concat_relu(x):
   axis = len(x.get_shape())-1
   return tf.nn.relu(tf.concat([x, -x], axis))

'''
   Regular relu
'''
def relu(x, name='relu'):
   return tf.nn.relu(x, name)

'''
   Tanh
'''
def tanh(x, name='tanh'):
   return tf.nn.tanh(x, name)

'''
   Sigmoid
'''
def sig(x, name='sig'):
   return tf.nn.sigmoid(x, name)

'''
   Self normalizing neural networks paper
   https://arxiv.org/pdf/1706.02515.pdf
'''
def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


###### end activation functions #########





