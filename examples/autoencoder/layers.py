#coding:utf8

import tensorflow as tf

from layer_utils import get_deconv2d_output_dims


def conv(input, name, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    assert(len(input_dims) == 4) # batch_size, height, width, num_channels_in
    assert(len(filter_dims) == 3) # height, width and num_channels out
    assert(len(stride_dims) == 2) # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    # Define a variable scope for the conv layer
    with tf.variable_scope(name) as scope:
        # Create filter weight variable
        weight = tf.Variable(tf.random_normal([filter_h, filter_w, num_channels_in, num_channels_out]), name='weight')

        # Create bias variable
        bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[num_channels_out]), name='bias')

        # Define the convolution flow graph
        conv = tf.nn.conv2d(input, weight, [1, stride_h, stride_w, 1], padding=padding)

        # Add bias to conv output
        output = tf.nn.bias_add(conv, bias, name='conv')

        # Apply non-linear (if asked) and return output
        if non_linear_fn:
            output = non_linear_fn(output)
        return output


def deconv(input, name, filter_dims, stride_dims, padding='SAME',
           non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    assert(len(input_dims) == 4) # batch_size, height, width, num_channels_in
    assert(len(filter_dims) == 3) # height, width and num_channels out
    assert(len(stride_dims) == 2) # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims
    # Let's step into this function
    output_dims = get_deconv2d_output_dims(input_dims,
                                           filter_dims,
                                           stride_dims,
                                           padding)
    # Define a variable scope for the deconv layer
    with tf.variable_scope(name) as scope:
        # Create filter weight variable
        # Note that num_channels_out and in positions are flipped for deconv.
        weight = tf.Variable(tf.random_normal([filter_h, filter_w, num_channels_out, num_channels_in], name='weight'))

        # Create bias variable
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[num_channels_out]), name='bias')

        # Define the deconv flow graph
        deconv = tf.nn.conv2d_transpose(input, weight, output_dims, [1, stride_h, stride_w, 1], padding)

        # Add bias to deconv output
        output = tf.nn.bias_add(deconv, bias, name='output')

        # Apply non-linearity (if asked) and return output
        if non_linear_fn:
            output = non_linear_fn(output, name='output')
        return output


def max_pool(input, name, filter_dims, stride_dims, padding='SAME'):
    assert(len(filter_dims) == 2) # filter height and width
    assert(len(stride_dims) == 2) # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    # Define the max pool flow graph and return output
    pool = tf.nn.max_pool(input,
                          ksize=[1, filter_h, filter_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding,
                          name=name)
    return pool


def fc(input, name, out_dim, non_linear_fn=tf.nn.relu):
    assert (type(out_dim) == int)

    # Define a variable scope for the FC layer
    with tf.variable_scope(name) as scope:
        input_dims = input.get_shape().as_list()
        # the input to the fc layer should be flattened
        if len(input_dims) == 4:
            # for eg. the output of a conv layer
            batch_size, input_h, input_w, num_channels = input_dims
            # ignore the batch dimension
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input, [batch_size, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input

        # Create weight variable
        weight = tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32), name='weight')

        # Create bias variable
        bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[out_dim]), name='bias')

        # Define FC flow graph
        output = tf.matmul(flat_input, weight) + bias

        # Apply non-linearity (if asked) and return output
        if non_linear_fn:
            output = non_linear_fn(output)

        return output