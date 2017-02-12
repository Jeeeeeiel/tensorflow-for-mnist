#! /usr/bin/python3
# __*__ coding: utf-8 __*__


import tensorflow as tf

NUM_CLASSES = 10
labels = [2, 3, 4, 5, 3, 2]
with tf.Session() as sess:
    batch_size = tf.size(labels)
    print(batch_size.eval())
    labels = tf.expand_dims(labels, 1)
    print(labels.eval())
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    print(indices.eval())
    concated = tf.concat(1, [indices, labels])
    print(concated.eval())
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    print(tf.pack([batch_size, NUM_CLASSES]).eval())
    print(onehot_labels.eval())

# x = [1, 2]
# with tf.Session():
#     print(tf.shape(x).eval())
#     print(tf.shape(tf.expand_dims(x, 1)).eval())
