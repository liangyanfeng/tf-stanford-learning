# coding:utf8

import tensorflow as tf

from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    conv1_output = conv(input, 'conv1', [3, 3, 1], [1, 1])
    # conv1_output = max_pool(conv1_output, 'conv1_pool', [3, 3], [2, 2])

    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    conv2_output = conv(conv1_output, 'conv2', [3, 3, 8], [1, 1])
    # conv2_output = max_pool(conv2_output, 'conv2_pool', [3, 3], [2, 2])

    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    conv3_output = conv(conv2_output, 'conv3', [3, 3, 8], [1, 1])
    # conv3_output = max_pool(conv3_output, 'conv3_pool', [3, 3], [2, 2])

    # FC: output_dim: 100, no non-linearity
    fc_output = fc(conv3_output, 'conv_fc', 100, None)

    return fc_output


def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    deconv_fc_output = fc(input, 'deconv_fc', 128)

    # Reshape to [batch_size, 4, 4, 8]
    batch_size = int(input.get_shape()[0])
    deconv_fc_output_reshape = tf.reshape(deconv_fc_output, shape=[batch_size, 4, 4, 8])

    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    deconv1_output = deconv(deconv_fc_output_reshape, 'deconv1', [3, 3, 8], [2, 2])

    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    deconv2_output = deconv(deconv1_output, 'deconv2', [8, 8, 1], [2, 2], padding='VALID')

    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    deconv3_output = deconv(deconv2_output, 'deconv3', [7, 7, 1], [1, 1], padding='VALID', non_linear_fn=tf.nn.sigmoid)

    return deconv3_output


def autoencoder(input_shape):
    # Define place holder with input shape
    input_image = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_image')

    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoding = encoder(input_image)
        # Pass encoding into decoder to obtain reconstructed image
        reconstructed = decoder(encoding)

        # Return input image (placeholder) and reconstructed image
        return input_image, reconstructed