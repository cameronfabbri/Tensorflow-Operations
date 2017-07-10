'''

   Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math


'''
   Converts a single image from [0,255] range to [-1,1]
'''
def preprocess(image):
   with tf.name_scope('preprocess'):
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return (image/127.5)-1.0

'''
   Converts a single image from [-1,1] range to [0,255]
'''
def deprocess(image):
   with tf.name_scope('deprocess'):
      return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)
   
'''
   Converts a batch of images from [-1,1] range to [0,255]
'''
def batch_convert2int(images):
  return tf.map_fn(deprocess, images, dtype=tf.uint8)

'''
   Converts a batch of images from [0,255] to [-1,1]
'''
def batch_convert2float(images):
  return tf.map_fn(preprocess, images, dtype=tf.float32)


'''
   Convert an image from RGB color range to LAB
   https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py#L155
'''
def rgb_to_lab(srgb):
   with tf.name_scope('rgb_to_lab'):
      srgb_pixels = tf.reshape(srgb, [-1, 3])
      with tf.name_scope('srgb_to_xyz'):
         linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
         exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
         rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
         rgb_to_xyz = tf.constant([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334], # R
            [0.357580, 0.715160, 0.119193], # G
            [0.180423, 0.072169, 0.950227], # B
         ])
         xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

      # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
      with tf.name_scope('xyz_to_cielab'):
         # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

         # normalize for D65 white point
         xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

         epsilon = 6/29
         linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
         exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
         fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

         # convert to lab
         fxfyfz_to_lab = tf.constant([
         #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
         ])
         lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

      return tf.reshape(lab_pixels, tf.shape(srgb))

'''
   Convert LAB colorspace to RGB
   https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py#L196
'''
def lab_to_rgb(lab):
   with tf.name_scope('lab_to_rgb'):
      lab = check_image(lab)
      lab_pixels = tf.reshape(lab, [-1, 3])
      # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
      with tf.name_scope('cielab_to_xyz'):
         # convert to fxfyfz
         lab_to_fxfyfz = tf.constant([
            #   fx      fy        fz
            [1/116.0, 1/116.0,  1/116.0], # l
            [1/500.0,     0.0,      0.0], # a
            [    0.0,     0.0, -1/200.0], # b
         ])
         fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

         # convert to xyz
         epsilon = 6/29
         linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
         exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
         xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

         # denormalize for D65 white point
         xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

      with tf.name_scope('xyz_to_srgb'):
         xyz_to_rgb = tf.constant([
            #     r           g          b
            [ 3.2404542, -0.9692660,  0.0556434], # x
            [-1.5371385,  1.8760108, -0.2040259], # y
            [-0.4985314,  0.0415560,  1.0572252], # z
         ])
         
         rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
         # avoid a slightly negative number messing up the conversion
         rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
         linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
         exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
         srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

      return tf.reshape(srgb_pixels, tf.shape(lab))

