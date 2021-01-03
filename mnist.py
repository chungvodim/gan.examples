import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

# Prepare dataset
BUFFER_SIZE = 60000
BATCH_SIZE = 256
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) # Batch and shuffle the data

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    # https://keras.io/api/layers/core_layers/dense/
    # Just your regular densely-connected NN layer.
    # tf.keras.layers.Dense(
    #     units,: Positive integer, dimensionality of the output space.
    #     activation=None,: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    #     use_bias=True,: Boolean, whether the layer uses a bias vector.
    #     kernel_initializer="glorot_uniform",: Initializer for the kernel weights matrix.
    #     bias_initializer="zeros",: Initializer for the bias vector.
    #     kernel_regularizer=None,: Regularizer function applied to the kernel weights matrix.
    #     bias_regularizer=None,: Regularizer function applied to the bias vector.
    #     activity_regularizer=None,: Regularizer function applied to the output of the layer (its "activation").
    #     kernel_constraint=None,: Constraint function applied to the kernel weights matrix.
    #     bias_constraint=None,: Constraint function applied to the bias vector.
    #     **kwargs
    # )
    model.add(layers.BatchNormalization())
    # https://keras.io/api/layers/normalization_layers/batch_normalization/
    # Layer that normalizes its inputs.
    # tf.keras.layers.BatchNormalization(
    #     axis=-1,
    #     momentum=0.99,
    #     epsilon=0.001,
    #     center=True,
    #     scale=True,
    #     beta_initializer="zeros",
    #     gamma_initializer="ones",
    #     moving_mean_initializer="zeros",
    #     moving_variance_initializer="ones",
    #     beta_regularizer=None,
    #     gamma_regularizer=None,
    #     beta_constraint=None,
    #     gamma_constraint=None,
    #     renorm=False,
    #     renorm_clipping=None,
    #     renorm_momentum=0.99,
    #     fused=None,
    #     trainable=True,
    #     virtual_batch_size=None,
    #     adjustment=None,
    #     name=None,
    #     **kwargs
    # )
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
    # https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
    # https://keras.io/api/layers/convolution_layers/convolution2d_transpose/
    # Transposed convolutional layer is usually carried out for upsampling
    # tf.keras.layers.Conv2DTranspose(
    #     filters,: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    #     kernel_size,: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
    #     strides=(1, 1),: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
    #     padding="valid",: "valid" means no padding."same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
    #     output_padding=None,
    #     data_format=None,
    #     dilation_rate=(1, 1),
    #     activation=None,
    #     use_bias=True,
    #     kernel_initializer="glorot_uniform",
    #     bias_initializer="zeros",
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     bias_constraint=None,
    #     **kwargs
    # )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model