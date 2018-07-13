import tensorflow as tf


def conv(x, filter_shape, stride):
    filters = tf.Variable(tf.truncated_normal(filter_shape,
                                               mean=0.0, stddev=1.0))
    return tf.nn.conv2d(x, filter=filters, strides=[1, stride, stride, 1],
                        padding="SAME")


def res_unit(x, filter_size, in_dimension, out_dimension, stride):
    # TODO: should check how to do batch normalization correctly
    prev_norm = tf.layers.batch_normalization(x)
    prev_out = tf.nn.relu(prev_norm)

    conv_1 = conv(prev_out, [filter_size, filter_size, in_dimension,
                             out_dimension], stride)
    # TODO: should check how to do batch normalization correctly
    norm_1 = tf.layers.batch_normalization(conv_1)
    layer_1 = tf.nn.relu(norm_1)

    conv_2 = conv(layer_1, [filter_size, filter_size, out_dimension,
                            out_dimension], 1)

    if in_dimension != out_dimension:
        # TODO: check the right way for size reduction
        size_reduction = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding='VALID')
        pad = (out_dimension - in_dimension) // 2
        x_padded = tf.pad(size_reduction, [[0, 0], [0, 0], [0, 0], [pad, pad]])

    else:
        x_padded = x

    return tf.add(conv_2, x_padded)


def cifar_resnet(n, images):
    FILTER_SIZE = 3
    LAYER1_DIMENSION = 16
    LAYER2_DIMENSION = 32
    LAYER3_DIMENSION = 64

    level1 = conv(images, [3, 3, 3, 16], 1)

    for _ in range(n):
        level1 = res_unit(level1, FILTER_SIZE, LAYER1_DIMENSION,
                          LAYER1_DIMENSION, 1)

    level2 = res_unit(level1, FILTER_SIZE, LAYER1_DIMENSION,
                      LAYER2_DIMENSION, 2)

    for _ in range(n-1):
        level2 = res_unit(level2, FILTER_SIZE, LAYER2_DIMENSION,
                          LAYER2_DIMENSION, 1)

    level3 = res_unit(level2, FILTER_SIZE, LAYER2_DIMENSION,
                      LAYER3_DIMENSION, 2)

    for _ in range(n - 1):
        level3 = res_unit(level3, FILTER_SIZE, LAYER3_DIMENSION,
                          LAYER3_DIMENSION, 1)

    norm = tf.layers.batch_normalization(level3)
    conv_out = tf.nn.relu(norm)
    global_pool = tf.reduce_mean(conv_out, [1, 2])

    batch_size = images.get_shape().as_list()[0]
    out = tf.layers.dense(inputs=tf.reshape(global_pool, [batch_size, -1]),
                          units=10,
                          kernel_initializer=tf.contrib.layers.xavier_initializer())

    return out
