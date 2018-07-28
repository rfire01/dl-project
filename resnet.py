"""Implemention of residual network."""
import os
import cv2
import numpy
import pickle
import tensorflow as tf


def conv(x, filter_shape, stride):
    """Activate convolution on x

    Args:
        x (tf.tensor): the input for the convolution.
        filter_shape (list): the shape of the filter [dim1, dim2, in_dimension,
            out_dimension]
        stride (number): the stride to use for the filter.

    Returns:
        tf.tensor. output of the convolution.
    """
    filters = tf.Variable(tf.truncated_normal(filter_shape,
                                              mean=0.0, stddev=1.0))
    return tf.nn.conv2d(x, filter=filters, strides=[1, stride, stride, 1],
                        padding="SAME")


def normalize_batch(x):
    """Batch normalization for a 4-D tensor

    Args:
        x (tf.tensor): the input for the convolution.

    Returns:
        tf.tensor. the result of batch normalization.
    """
    mean, var = tf.nn.moments(x, axes=[0, 1, 2])
    constant_size = x.get_shape().as_list()[3]
    offset = tf.Variable(tf.zeros([constant_size]))
    scale = tf.Variable(tf.ones([constant_size]))
    batch_norm = tf.nn.batch_normalization(x, mean, var, offset, scale,
                                           0.001)
    return batch_norm


def res_unit(x, filter_size, in_dimension, out_dimension, stride, enable):
    """Create a residual unit of 2 layers.

    Args:
        x (tf.tensor): the input for the convolution.
        filter_size (number): the size of a square (n x n) filter.
        in_dimension (number): number of channels of x.
        out_dimension (number): number of channels of the residual unit.
        stride (number): the stride to use for the filter.
        enable (bool): create residual unit or regular 2 layers.

    Returns:
        tf.tensor. output of the residual unit.
    """
    prev_norm = normalize_batch(x)
    prev_out = tf.nn.relu(prev_norm)

    conv_1 = conv(prev_out, [filter_size, filter_size, in_dimension,
                             out_dimension], stride)
    norm_1 = normalize_batch(conv_1)
    layer_1 = tf.nn.relu(norm_1)

    conv_2 = conv(layer_1, [filter_size, filter_size, out_dimension,
                            out_dimension], 1)

    if in_dimension != out_dimension:
        size_reduction = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding='VALID')
        pad = (out_dimension - in_dimension) // 2
        x_padded = tf.pad(size_reduction, [[0, 0], [0, 0], [0, 0], [pad, pad]])

    else:
        x_padded = x

    if enable:
        return tf.add(conv_2, x_padded)

    else:
        return conv_2


class cifar_resnet(object):
    """Residual network for the CIFAR-10 database.

    Attributes:
        FILTER_SIZE (number):  the size of the filters between layers.
        LAYER1_DIMENSION (number): number of channels for first part of the
            network.
        LAYER2_DIMENSION (number): number of channels for second part of the
            network.
        LAYER3_DIMENSION (number): number of channels for third part of the
            network.
        CHECKPOINTS (list): at what iterations the network change the
            learning rate.
        CHECKPOINT_LEARNING_RATE (list): the learning rate values for each
            of the checkpoints.
    """
    FILTER_SIZE = 3
    LAYER1_DIMENSION = 16
    LAYER2_DIMENSION = 32
    LAYER3_DIMENSION = 64
    CHECKPOINTS = [20000, 48000]
    CHECKPOINT_LEARNING_RATE = [0.01, 0.001, 0.0001]

    def __init__(self, n, enable=True):
        self.step = tf.Variable(initial_value=0, trainable=False)
        self.learning_rate = tf.train.piecewise_constant(self.step,
                                                         self.CHECKPOINTS,
                                                         self.CHECKPOINT_LEARNING_RATE)

        self.images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [None, 10])

        level1 = conv(self.images, [3, 3, 3, 16], 1)

        index = 1

        for _ in range(n):
            level1 = res_unit(level1, self.FILTER_SIZE, self.LAYER1_DIMENSION,
                              self.LAYER1_DIMENSION, 1, enable)
            index += 1

        level2 = res_unit(level1, self.FILTER_SIZE, self.LAYER1_DIMENSION,
                          self.LAYER2_DIMENSION, 2, enable)
        index += 1

        for _ in range(n - 1):
            level2 = res_unit(level2, self.FILTER_SIZE, self.LAYER2_DIMENSION,
                              self.LAYER2_DIMENSION, 1, enable)
            index += 1

        level3 = res_unit(level2, self.FILTER_SIZE, self.LAYER2_DIMENSION,
                          self.LAYER3_DIMENSION, 2, enable)
        index += 1

        for _ in range(n - 1):
            level3 = res_unit(level3, self.FILTER_SIZE, self.LAYER3_DIMENSION,
                              self.LAYER3_DIMENSION, 1, enable)
            index += 1

        normed = normalize_batch(level3)
        conv_out = tf.nn.relu(normed)
        global_pool = tf.reduce_mean(conv_out, [1, 2])

        # Adding weight decay to the output layer
        w_regularize = tf.contrib.layers.l2_regularizer(scale=0.0001)
        fc_w = tf.get_variable(name="w_out", shape=[self.LAYER3_DIMENSION, 10],
                               initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                               regularizer=w_regularize)

        b_regularize = tf.contrib.layers.l2_regularizer(scale=0.0001)
        b = tf.get_variable(name="b_out", shape=[10],
                            initializer=tf.zeros_initializer(),
                            regularizer=b_regularize)
        self.output = tf.nn.softmax(tf.matmul(global_pool, fc_w) + b)

        self.loss = - tf.reduce_sum(self.labels * tf.log(self.output))
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        self.train_optimizer = optimizer.minimize(self.loss,
                                                  global_step=self.step)


def label_to_onehot(labels):
    """Conver labels to one-hot vector.

    Args:
        labels (numpy.array): array of labels for cifar-10 database.

    Returns:
        numpy.array. original labels converted to one-hot vector.
    """
    CLASS_NUM = 10

    onehot_labels = []
    for label in labels:
        onehot = numpy.zeros(CLASS_NUM)
        onehot[label] = 1
        onehot_labels.append(onehot)

    return numpy.array(onehot_labels)


def input_augmentation(images, pad=4):
    """Augmentation of images.

    Args:
        images (numpy.array): the original images.
        pad (number): how much pad add to the image before the augmentation.

    Return:
        numpy.array. the images after augmentation.
    """
    IMAGE_SIZE = 32

    pad_width = ((0, 0), (pad, pad), (pad, pad), (0, 0))
    padded_images = numpy.pad(images, pad_width, mode='constant',
                              constant_values=0)

    augmated_images = numpy.zeros_like(images)
    for image_index in range(len(images)):
        x_offset = numpy.random.randint(0, pad * 2)
        y_offset = numpy.random.randint(0, pad * 2)
        cropped_image = padded_images[image_index][x_offset:
                                                   x_offset + IMAGE_SIZE,
                                                   y_offset:
                                                   y_offset + IMAGE_SIZE, :]

        if numpy.random.randint(2) == 0:
            augmated_image = cv2.flip(cropped_image, 1)

        else:
            augmated_image = cropped_image

        numpy.copyto(augmated_images[image_index], augmated_image)

    return augmated_images


def evaluate(x, y, net, session):
    """Evaluate the network.

    Args:
        x (numpy.array): input images.
        y (numpy.array): correct labels for x.
        net (cifar_resnet): the network to evaluate.
        session (tf.session): session to use for evaluation.

    Returns:
         (number, number). the accuracy and loss of the network using x as
            input and y as correct ouput.
    """
    max_size = 100

    total_acc = 0
    total_loss = 0
    split_amount = len(x) / max_size
    for index in range(int(split_amount)):
        batch_x = x[max_size * index: max_size * (index + 1)]
        batch_y = y[max_size * index: max_size * (index + 1)]

        pred_Y = session.run(net.output, {net.images: batch_x})
        pred_Y = numpy.argmax(pred_Y, 1)

        correct_num = 0
        for i, j in zip(pred_Y, batch_y):
            if i == numpy.argmax(j):
                correct_num += 1

        total_acc += 1.0 * correct_num / max_size

        total_loss += session.run(net.loss, {net.images: batch_x,
                                      net.labels: batch_y})

    return total_acc / split_amount, total_loss


def main():
    ITERATION_AMOUNT = 64000
    CIFAR_BATCH_SIZE = 128

    # download the cifar-10 data.
    (train_x, train_y), \
    (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    # convert output labels to one-hot vectors.
    train_y = label_to_onehot(train_y)
    test_y = label_to_onehot(test_y)

    # shuffle the data
    shuffled_indices = numpy.random.permutation(len(train_x))
    train_x = train_x[shuffled_indices]
    train_y = train_y[shuffled_indices]

    resnet = cifar_resnet(3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_counter = 0
    epoch = 1
    # running 64k iteration, a batch at each iteration.
    for iteration in range(ITERATION_AMOUNT):
        if iteration % 100 == 0:
            print("iteration ", iteration)
        batch_x = train_x[batch_counter: batch_counter + CIFAR_BATCH_SIZE]
        batch_x = input_augmentation(batch_x)
        batch_y = train_y[batch_counter: batch_counter + CIFAR_BATCH_SIZE]

        sess.run(resnet.train_optimizer,
                 {resnet.images: batch_x, resnet.labels: batch_y})

        batch_counter += CIFAR_BATCH_SIZE
        # evaluate the network after each epoch ends.
        if batch_counter >= len(train_x):
            batch_counter = 0

            print("epoch ", epoch)
            test_acc, test_loss = evaluate(test_x, test_y, resnet, sess)
            print("test error - ", 1 - test_acc)
            print("test loss - ", test_loss)
            epoch += 1


if __name__ == "__main__":
    main()
