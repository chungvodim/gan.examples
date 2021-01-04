# 1. Chuẩn bị dữ liệu
# 2. Xây dựng network
# 3. Chọn thuật toán cập nhật nghiệm, xây dựng loss và phương pháp đánh giá mô hình
# 4. Huấn luyện mô hình.
# 5. Đánh giá mô hình

import tensorflow as tf
import glob
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

matplotlib.interactive(True)
# Prepare dataset
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # 127.5 = 255/2, Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) # Batch and shuffle the data

# Build Models
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

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Build Loss Functions
# z: noise
# G(z): generator's output (fake instance)
# D(G(z)): critic's output for fake instance
# D(x): critic's output for real instance

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)# This method returns a helper function to compute cross entropy loss

# Generator loss: D(G(z))
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)# tf.ones_like: Creates a tensor of all ones that has the same shape as the input.

# Discriminator loss: D(x) - D(G(z))
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)# tf.zeros_like: Creates a tensor with all elements set to zero.
    total_loss = real_loss + fake_loss
    return total_loss

# make generator and test
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# make discriminator and test
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# Optimizers
# https://www.jeremyjordan.me/nn-learning-rate/
# If your learning rate is set too low, training will progress very slowly as you are making very tiny updates to the weights in your network.
# If your learning rate is set too high, it can cause undesirable divergent behavior in your loss function.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
                                 
# Define the training loop
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # tf.GradientTape: Record operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Train the model
train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Create a GIF
# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
  
import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)