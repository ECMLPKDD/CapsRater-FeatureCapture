import argparse

import tensorflow as tf
from keras import utils
from sklearn.metrics import accuracy_score
from bert_serving.client import BertClient
from sklearn.metrics import precision_recall_fscore_support

from config import *
from features import *
from model import *
from loss import cross_entropy
from data import get_data, BatchGenerator
from eval import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt_nbr", type=int, default=1, help="Options: 1, 2, 3, 4, 5, 6, 7, 8"
)

args = parser.parse_args()

tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)


def main():

    train, train_label, dev, dev_label, test, test_label, num_classes = get_data(args.prompt_nbr)

    vectorizer = BertClient(ip="localhost", port=8190)

    bert_vec_train = vectorizer.encode(train.tolist())
    bert_vec_dev = vectorizer.encode(dev.tolist())
    bert_vec_test = vectorizer.encode(test.tolist())

    with tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()


    x = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENT], name="input_x")
    y = tf.placeholder(tf.int64, [BATCH_SIZE, num_classes], name="input_y")
    is_training = tf.placeholder_with_default(False, shape=())
    learning_rate = tf.placeholder(dtype='float32')

    bert_vec_train = np.array(bert_vec_train,dtype=np.float32)
    w_1 = tf.Variable(bert_vec_train, trainable = False)
    input_embedding = tf.nn.embedding_lookup(w_1, x)
    input_embedding = input_embedding[...,tf.newaxis]

    poses, activations = capsule_model(input_embedding, num_classes)

    train_features = get_all_features(train)
    dev_features = get_all_features(dev)
    test_features = get_all_features(test)

    feature_output = feature_capture_model(train_features, train_label, test_features, [(dev_features, dev_label)])

    h1 = tf.reduce_mean([activations, feature_output], 0)

    output = tf.compat.v1.layers.dense( h1, num_classes, activation = tf.nn.softmax, use_bias=True, kernel_initializer=None,bias_initializer=tf.zeros_initializer())

    loss = cross_entropy(y, output)

    y_pred = tf.argmax(output, axis=1, name="y_probability")
    correct = tf.equal(tf.argmax(y, axis=1), y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, name="training_op")
    gradients, variables = zip(*optimizer.compute_gradients(loss))

    grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                  for g in gradients if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]
    with tf.control_dependencies(grad_check):
        training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    sess = tf.InteractiveSession()

    n_iterations_per_epoch = len(bert_vec_train) // BATCH_SIZE
    n_iterations_test = len(bert_vec_dev) // BATCH_SIZE
    n_iterations_dev = len(bert_vec_test) // BATCH_SIZE

    batch_train = BatchGenerator(bert_vec_train, train_label, BATCH_SIZE, 0)
    batch_dev = BatchGenerator(bert_vec_dev, dev_label, BATCH_SIZE, 0)
    batch_test = BatchGenerator(bert_vec_test, test_label, BATCH_SIZE, 0, is_shuffle=False)

    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(NUM_EPOCHS):
        for iteration in range(1, n_iterations_per_epoch + 1):
            x_batch, y_batch = batch_train.next()
            y_batch = utils.to_categorical(y_batch, num_classes)
            _, loss_train, _, _ = sess.run(
                [training_op, loss, output, poses],
                feed_dict={x: x_batch[:,:MAX_SENT],
                           y: y_batch,
                           is_training: True,
                           learning_rate:LEARNING_RATE})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")
        loss_vals, acc_vals = [], []
        for iteration in range(1, n_iterations_dev + 1):
            x_batch, y_batch = batch_dev.next()
            y_batch = utils.to_categorical(y_batch, num_classes)
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={x: x_batch[:,:MAX_SENT],
                               y: y_batch,
                               is_training: False})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val, acc_val = np.mean(loss_vals), np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.1f}%  Loss: {:.4f}".format(
            epoch + 1, acc_val * 100, loss_val))

        pred_label, true_label = [], []
        for iteration in range(1, n_iterations_test + 1):
            x_batch, y_batch = batch_test.next()
            prob = sess.run([output],
                    feed_dict={x:x_batch[:,:MAX_SENT],
                               is_training: False})
            pred_label = pred_label + prob[0].tolist()
            true_label = true_label + y_batch.tolist()

        true_label = np.array(true_label)
        pred_probs = np.array(pred_label)

        QWK = quadratic_weighted_kappa(true_label,pred_probs)

        print ('QWK: %.3f' % QWK)


if __name__ == "__main__":
    main()
