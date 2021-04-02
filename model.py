from __future__ import division, print_function, unicode_literals

import xgboost as xgb
import tensorflow as tf

from utils import _conv2d_wrapper
from layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer


def capsule_model(X, num_classes):
    with tf.variable_scope("capsule_" + str(3)):
        nets = _conv2d_wrapper(
            X,
            shape=[3, 300, 1, 32],
            strides=[1, 2, 1, 1],
            padding="VALID",
            add_bias=True,
            activation_fn=tf.nn.relu,
            name="conv1",
        )
        tf.logging.info("output shape: {}".format(nets.get_shape()))
        nets = capsules_init(
            nets,
            shape=[1, 1, 32, 16],
            strides=[1, 1, 1, 1],
            padding="VALID",
            pose_shape=16,
            add_bias=True,
            name="primary",
        )
        nets = capsule_conv_layer(
            nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name="conv2"
        )
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, "fc2")
    return poses, activations


def feature_capture_model(x_train, y_train, x_test, dev_set):
    xgb = xgb.XGBClassifier(learning_rate =0.0001,
                             n_estimators=1000,
                             max_depth=6,
                             objective= 'multi:softprob',
                             seed=43)
    xgb.fit(x_train, y_train, eval_set=dev_set)
    preds = xgb.predict(x_test)
    return preds