import tensorflow as tf


def cross_entropy(y, preds):
    y = tf.argmax(y, axis=1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=y)
    loss = tf.reduce_mean(loss)
    return loss
