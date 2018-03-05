import os

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import get_config, print_usage
from utils.cifar10 import load_data
from utils.features import extract_h_histogram, extract_hog
from utils.preprocess import normalize

data_dir = "/Users/kwang/Downloads/cifar-10-batches-py"


class MyNetwork(object):
    """Network class """

    def __init__(self, x_shp, config):

        self.config = config

        self.x_shp = x_shp

        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        self.x_in = tf.placeholder(tf.float32, shape=(None, self.x_shp[1]))
        self.y_in = tf.placeholder(tf.int64, shape=(None, ))

    def _build_preprocessing(self):

        with tf.variable_scope("Normalization", reuse=tf.AUTO_REUSE):

            self.n_mean_in = tf.placeholder(
                tf.float32, shape=(1, self.x_shp[1]))
            self.n_range_in = tf.placeholder(
                tf.float32, shape=(1, self.x_shp[1]))

            self.n_mean = tf.get_variable(name="n_mean", dtype=tf.float32, shape=(1, self.x_shp[1]), initializer=tf.zeros_initializer(), trainable=False)
            self.n_range = tf.get_variable(name="n_range", dtype=tf.float32, shape=(1, self.x_shp[1]), initializer=tf.zeros_initializer(), trainable=False)

            apply_mean = tf.assign(self.n_mean, self.n_mean_in)
            apply_range = tf.assign(self.n_range, self.n_range_in)
            self.n_assign_op = tf.group(apply_mean, apply_range)

    def _build_model(self):

        num_unit = self.config.num_unit
        kernel_initializer = tf.keras.initializers.he_normal()

        with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):

            cur_in = (self.x_in - self.n_mean) / self.n_range

            if(self.config.activ_type == "tanh"):
                kernel_initializer = tf.glorot_normal_initializer()
            cur_in = tf.layers.dense(inputs=cur_in, units=num_unit, kernel_initializer=kernel_initializer)

            if self.config.activ_type == "relu":
                cur_in = tf.nn.relu(cur_in)
            else:
                cur_in = tf.nn.tanh(cur_in)

            for _i in range(self.config.num_hidden):

                cur_in = tf.layers.dense(inputs=cur_in, units=num_unit, kernel_initializer=kernel_initializer)

                if self.config.activ_type == "relu":
                    cur_in = tf.nn.relu(cur_in)
                else:
                    cur_in = tf.nn.tanh(cur_in)

            self.logits = tf.layers.dense(inputs=cur_in, units=self.config.num_class, kernel_initializer=kernel_initializer)

            self.kernels_list = [
                _v for _v in tf.trainable_variables() if "kernel" in _v.name]

    def _build_loss(self):


        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            self.onehot = tf.one_hot(indices=self.y_in, depth=self.config.num_class)

            loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot, logits=self.logits)

            self.loss = tf.reduce_mean(loss_i)

            l2_loss = tf.add_n([
                tf.reduce_sum(_v**2) for _v in self.kernels_list])
            self.loss += self.config.reg_lambda * l2_loss

            tf.summary.scalar("loss", self.loss)

    def _build_optim(self):


        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):

            self.global_step = tf.get_variable(name="global_step", dtype=tf.int32, shape=[], initializer=tf.zeros_initializer())

            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss, global_step=self.global_step)
            
    def _build_eval(self):


        with tf.variable_scope("Eval", tf.AUTO_REUSE):

            self.pred = tf.argmax(self.logits, 1)

            self.acc = tf.divide(tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.y_in))), tf.to_float(tf.shape(self.x_in)[0]), name="acc")

            self.accuracy_scalar = tf.summary.scalar("accuracy", self.acc)

    def _build_summary(self):

        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):

        self.summary_tr = tf.summary.FileWriter("{}/train".format(self.config.log_dir))
        self.summary_va = tf.summary.FileWriter("{}/valid".format(self.config.log_dir))

        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()

        self.save_file_cur = "{}/model".format(self.config.log_dir)

        self.save_file_best = "{}/model".format(self.config.save_dir)

    def train(self, x_tr, y_tr, x_va, y_va):

        print("Training data before: mean {}, std {}, min {}, max {}".format(
            x_tr.mean(), x_tr.std(), x_tr.min(), x_tr.max()
        ))

        _, x_tr_mean, x_tr_range = normalize(x_tr)

        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:
            print("Initializing...")

            init = tf.global_variables_initializer()
            sess.run(init)

            sess.run(
                self.n_assign_op,
                feed_dict={
                    self.n_mean_in: x_tr_mean,
                    self.n_range_in: x_tr_range,
                }
            )

            print("Testing...")
            res = sess.run(
                fetches = {
                    "summary": self.summary_op,
                    "global_step": self.global_step,
                },
                feed_dict={
                    self.x_in: x_va,
                    self.y_in: y_va,
                }
            )

            self.summary_va.add_summary(res["summary"], global_step=res["global_step"])

            print("Training...")
            batch_size = config.batch_size
            num_epoch = config.num_epoch
            num_batch = len(x_tr) // batch_size
            best_acc = 0

            for idx_epoch in trange(num_epoch):

                ind_data = np.random.permutation(len(x_tr))

                for idx_batch in range(num_batch):

                    ind_cur = ind_data[
                        batch_size * idx_batch:batch_size * (idx_batch + 1)
                    ]

                    x_b = np.array([x_tr[_i] for _i in ind_cur])
                    y_b = np.array([y_tr[_i] for _i in ind_cur])

                    res = sess.run(
                        fetches = {
                            "accuracy": self.acc,
                            "optim": self.optim,
                            "summary": self.summary_op,
                            "global_step": self.global_step,
                        },
                        feed_dict={
                            self.x_in: x_b,
                            self.y_in: y_b,
                        }
                    )

                    self.summary_tr.add_summary(res["summary"], global_step=res["global_step"])

                self.summary_tr.flush()

                res = sess.run(
                        fetches = {
                            "accuracy": self.acc,
                            "summary": self.summary_op,
                            "global_step": self.global_step,
                        },
                        feed_dict={
                            self.x_in: x_va,
                            self.y_in: y_va,
                        }
                )

                self.summary_va.add_summary(res["summary"], global_step=res["global_step"])

                self.summary_va.flush()

                self.saver_cur.save(sess, self.save_file_cur, global_step=self.global_step, write_meta_graph=False)

                if res["accuracy"] > best_acc:
                    best_acc = res["accuracy"]

                    self.saver_best.save(sess, self.save_file_best, write_meta_graph=False)

    def test(self, x_te, y_te):


        with tf.Session() as sess:

            latest_checkpoint = tf.train.latest_checkpoint(self.config.save_dir)
            if latest_checkpoint is not None:
                print("restored")
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                    )
            else:
                print("restore failed")


            res = sess.run(
                    fetches = {
                        "accuracy": self.acc,
                    },
                    feed_dict={
                        self.x_in: x_te,
                        self.y_in: y_te,
                    }
            )

            print("Test accuracy with the best model is {}".format(
                res["accuracy"]))


def main(config):


    print("Reading training data...")
    data_trva, y_trva = load_data(config.data_dir, "train")

    print("Reading test data...")
    data_te, y_te = load_data(config.data_dir, "test")

    print("Extracting Features...")
    if config.feature_type == "hog":
        # HOG features
        x_trva = extract_hog(data_trva)
        x_te = extract_hog(data_te)
    elif config.feature_type == "h_histogram":
        # Hue Histogram features
        x_trva = extract_h_histogram(data_trva)
        x_te = extract_h_histogram(data_te)
    elif config.feature_type == "rgb":
        # raw RGB features
        x_trva = data_trva.astype(float).reshape(len(data_trva), -1)
        x_te = data_te.astype(float).reshape(len(data_te), -1)

    # Randomly shuffle data and labels
    idx_shuffle = np.random.permutation(len(x_trva))
    x_trva = x_trva[idx_shuffle]
    y_trva = y_trva[idx_shuffle]

    x_trva = x_trva.astype("float32")
    y_trva = y_trva.astype("int64")

    num_tr = int(len(x_trva) * 0.8)

    x_tr = x_trva[:num_tr]
    x_va = x_trva[num_tr:]
    y_tr = y_trva[:num_tr]
    y_va = y_trva[num_tr:]


    mynet = MyNetwork(x_tr.shape, config)


    mynet.train(x_tr, y_tr, x_va, y_va)


    mynet.test(x_te, y_te)


if __name__ == "__main__":


    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
    
