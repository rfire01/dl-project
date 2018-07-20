import os
import numpy
import pickle
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


class cifar_resnet:
    FILTER_SIZE = 3
    LAYER1_DIMENSION = 16
    LAYER2_DIMENSION = 32
    LAYER3_DIMENSION = 64

    def __init__(self,n ):
        self.images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [None, 10])

        level1 = conv(self.images, [3, 3, 3, 16], 1)

        for _ in range(n):
            level1 = res_unit(level1, self.FILTER_SIZE, self.LAYER1_DIMENSION,
                              self.LAYER1_DIMENSION, 1)

        level2 = res_unit(level1, self.FILTER_SIZE, self.LAYER1_DIMENSION,
                          self.LAYER2_DIMENSION, 2)

        for _ in range(n-1):
            level2 = res_unit(level2, self.FILTER_SIZE, self.LAYER2_DIMENSION,
                              self.LAYER2_DIMENSION, 1)

        level3 = res_unit(level2, self.FILTER_SIZE, self.LAYER2_DIMENSION,
                          self.LAYER3_DIMENSION, 2)

        for _ in range(n - 1):
            level3 = res_unit(level3, self.FILTER_SIZE, self.LAYER3_DIMENSION,
                              self.LAYER3_DIMENSION, 1)

        norm = tf.layers.batch_normalization(level3)
        conv_out = tf.nn.relu(norm)
        global_pool = tf.reduce_mean(conv_out, [1, 2])

        fc_w = tf.Variable(tf.truncated_normal([self.LAYER3_DIMENSION, 10],
                                                   mean=0.0, stddev=1.0))
        b = tf.Variable(tf.zeros([10]))
        self.output = tf.nn.softmax(tf.matmul(global_pool, fc_w) + b)

        loss = - tf.reduce_sum(self.labels + tf.log(self.output))
        optimizer = tf.train.MomentumOptimizer(0.001, momentum=0.9)
        self.train_optimizer = optimizer.minimize(loss)


def load_cifar_data(files):
    CHANNELS = 3
    CLASS_NUM = 10
    IAMGE_SIZE = 32
    BATCH_SIZE = 10000
    CIFAR_FOLDER = 'cifar-10-batches-py'

    data = []
    labels = []

    files_path = [os.path.join(CIFAR_FOLDER, file) for file in files]
    for file in files_path:
        with open(file, 'rb')as f:
            imported_data = pickle.load(f, encoding='latin1')
            extracted_data = imported_data['data']
            data.append(extracted_data.reshape(BATCH_SIZE, CHANNELS,
                                               IAMGE_SIZE, IAMGE_SIZE))
            labels.append(numpy.array(imported_data['labels']))

    all_data = numpy.concatenate(data, 0)
    all_data = all_data.transpose(0, 2, 3, 1)
    all_labels = numpy.concatenate(labels, 0)

    onehot_labels = []
    for label in all_labels:
        onehot = numpy.zeros(CLASS_NUM)
        onehot[label] = 1
        onehot_labels.append(onehot)

    return all_data, numpy.array(onehot_labels)


def main():
    CIFAR_TRAIN_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                         'data_batch_4', 'data_batch_5']
    CIFAR_TEST_FILES = ['test_batch']

    ITERATION_AMOUNT = 64000
    CIFAR_BATCH_SIZE = 128

    train_x, train_y = load_cifar_data(CIFAR_TRAIN_FILES)
    test_x, test_y = load_cifar_data(CIFAR_TEST_FILES)

    # shuffle the data
    shuffled_indices = numpy.random.permutation(len(train_x))
    train_x = train_x[shuffled_indices]
    train_y = train_y[shuffled_indices]

    resnet = cifar_resnet(3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_counter = 0
    # running 64k iteration, a batch at each iteration.
    # TODO: i'm not sure about how to split iteration\batch\epchos.
    for iteration in range(ITERATION_AMOUNT):
        batch_x = train_x[batch_counter: batch_counter + CIFAR_BATCH_SIZE]
        batch_y = train_y[batch_counter: batch_counter + CIFAR_BATCH_SIZE]

        sess.run(resnet.train_optimizer,
                 {resnet.images: batch_x, resnet.labels: batch_y})


        ## testing only 1 batch of test.
        ## TODO: do a better test
        batch_x = test_x[batch_counter: batch_counter + CIFAR_BATCH_SIZE]
        batch_y = test_y[batch_counter: batch_counter + CIFAR_BATCH_SIZE]
        pred_Y = sess.run(resnet.output, {resnet.images: batch_x})
        pred_Y = numpy.argmax(pred_Y, 1)

        correct_num = 0
        for i, j in zip(pred_Y, batch_y):
            if i == numpy.argmax(j):
                correct_num += 1

        print(1.0 * correct_num / 128.0)

        ######################################

        batch_counter += CIFAR_BATCH_SIZE
        if batch_counter >= len(train_x):
            batch_counter = 0


if __name__== "__main__":
    main()