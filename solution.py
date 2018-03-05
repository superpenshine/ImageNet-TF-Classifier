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
        """Initializer

        This function initializes the network and constructs all the
        computational workflow of our network. Everything you need to do for
        this function should already have been done for you.

        Parameters
        ----------
        x_shp : tuple of `int`
            Shape of your input data. We actually only use the last dimension
            in our implementation of our network

        config : Python namespace
            Configuration namespace.

        """

        self.config = config

        # Get shape
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
        """Build placeholders."""

        # Create Placeholders for inputs. NOTE: that we are having the first
        # dimension for these placeholders to be None so that it is determined
        # at runtime.
        self.x_in = tf.placeholder(tf.float32, shape=(None, self.x_shp[1]))
        self.y_in = tf.placeholder(tf.int64, shape=(None, ))

    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        with tf.variable_scope("Normalization", reuse=tf.AUTO_REUSE):
            # Create placeholders for saving mean, range to a TF variable for
            # easy save/load. Create these variables as well.
            self.n_mean_in = tf.placeholder(
                tf.float32, shape=(1, self.x_shp[1]))
            self.n_range_in = tf.placeholder(
                tf.float32, shape=(1, self.x_shp[1]))
            # TODO: Make the normalization as a TensorFlow variable. This is to
            # make sure we save it in the graph. `tf.get_variable`
            self.n_mean = tf.get_variable(name="n_mean", dtype=tf.float32, shape=(1, self.x_shp[1]), initializer=tf.zeros_initializer(), trainable=False)
            self.n_range = tf.get_variable(name="n_range", dtype=tf.float32, shape=(1, self.x_shp[1]), initializer=tf.zeros_initializer(), trainable=False)
            # TODO: Assign op to store this value to TF variable. See Lecture
            # 10 slides if you are unsure what I mean. You probably want to use
            # `tf.group` and `tf.assign` here.
            apply_mean = tf.assign(self.n_mean, self.n_mean_in)
            apply_range = tf.assign(self.n_range, self.n_range_in)
            self.n_assign_op = tf.group(apply_mean, apply_range)

    def _build_model(self):
        """Build our MLP network."""

        num_unit = self.config.num_unit
        kernel_initializer = tf.keras.initializers.he_normal()

        # Build the network (use tf.layers)
        with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
            # TODO: Normalize self.x_in using the above training-time
            # statistics, that is, self.n_mean and self.n_range. We will abuse
            # cur_in from now on to make coding a bit less painful.
            cur_in = (self.x_in - self.n_mean) / self.n_range
            # TODO: Input layer. This should be a `tf.layers.dense` layer that
            # takes `cur_in` as input. You should use either
            # `tf.keras.initializers.he_normal`, or
            # `tf.glorot_normal_initializer` to initialize this layer,
            # depending on the activation you use.
            if(self.config.activ_type == "tanh"):
                kernel_initializer = tf.glorot_normal_initializer()
            cur_in = tf.layers.dense(inputs=cur_in, units=num_unit, kernel_initializer=kernel_initializer)
            # TODO: An activation function depending on the
            # configuration. `tf.nn.relu` or `tf.nn.tanh`
            if self.config.activ_type == "relu":
                cur_in = tf.nn.relu(cur_in)
            else:
                cur_in = tf.nn.tanh(cur_in)
            # TODO: Hidden layers. You now probably see why we want to abuse
            # cur_in as input here.
            for _i in range(self.config.num_hidden):
                # TODO: Dense layer
                cur_in = tf.layers.dense(inputs=cur_in, units=num_unit, kernel_initializer=kernel_initializer)
                # TODO: Activation depending on the configuration
                if self.config.activ_type == "relu":
                    cur_in = tf.nn.relu(cur_in)
                else:
                    cur_in = tf.nn.tanh(cur_in)
            # TODO: Output layer. We need to now map to our output. For this
            # layer, we don't want any activations, since we want the output to
            # be `logits`. HINT: `self.config.num_class`.
            self.logits = tf.layers.dense(inputs=cur_in, units=self.config.num_class, kernel_initializer=kernel_initializer)

            # Get list of all weights in this scope. They are called "kernel"
            # in tf.layers.dense. This part is a bit advanced, so I will do it
            # for you. But try to understand what is going on. I will use this
            # list later to create the l2-regularizor loss.
            self.kernels_list = [
                _v for _v in tf.trainable_variables() if "kernel" in _v.name]

    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            # Create cross entropy loss.

            # TODO: First, let's create the onehot encoding with `self.y_in` to
            # make our lives easier. Try using `tf.one_hot` with
            # `self.config.num_class`
            self.onehot = tf.one_hot(indices=self.y_in, depth=self.config.num_class)
            # TODO: Get the cross entropy loss per sample. Note that you can
            # directly use the `tf.nn.softmax_cross_entropy_with_logits` (or
            # the v2 version if you are on more recent TF) to do that. All you
            # need to do is to provide the correct onehot and the correct
            # logits.
            loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot, logits=self.logits)
            # TODO: Now let's average them to get the loss of the batch with
            # `tf.reduce_mean`
            self.loss = tf.reduce_mean(loss_i)

            # Create l2 regularizer loss and add. See how I'm doing this here
            # :-) I like to do it this way, instead of using the regularizer
            # option in the parameters.
            l2_loss = tf.add_n([
                tf.reduce_sum(_v**2) for _v in self.kernels_list])
            self.loss += self.config.reg_lambda * l2_loss

            # Record summary for loss
            tf.summary.scalar("loss", self.loss)

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            # TODO: Create the global_step variable using
            # `tf.get_variable`. Recall Lecture 11.
            self.global_step = tf.get_variable(name="global_step", dtype=tf.int32, shape=[], initializer=tf.zeros_initializer())
            # TODO: We will use `tf.train.AdamOptimizer`. Use this to create
            # the optimize op, and store it at `self.optim` to be used
            # later. Also, apply a learning_rate of
            # `self.config.learning_rate`.
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss, global_step=self.global_step)
            
    def _build_eval(self):
        """Build the evaluation related ops"""

        with tf.variable_scope("Eval", tf.AUTO_REUSE):

            # TODO: Get predictions from your network. `tf.argmax` is your
            # friend. Apply this to the network output that we already defined
            # above.
            self.pred = tf.argmax(self.logits, 1)

            # TODO: Compute the accuracy of the model. When comparing labels
            # elemwise, use `tf.equal` instead of `==`. `==` will evaluate if
            # your Ops are identical Ops. You probably woule need to use
            # `tf.to_float` to convert your boolean tensors into floats in the
            # middle or TF will complain
            self.acc = tf.divide(tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.y_in))), tf.to_float(tf.shape(self.x_in)[0]), name="acc")

            # TODO: Record summary for accuracy. Use `tf.summary.scalar`
            self.accuracy_scalar = tf.summary.scalar("accuracy", self.acc)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # TODO: Create summary writers (one for train, one for validation). We
        # want each of them to save logs as "train" and "valid" sub directories
        # under `self.config.log_dir`.
        self.summary_tr = tf.summary.FileWriter("{}/train".format(self.config.log_dir))
        self.summary_va = tf.summary.FileWriter("{}/valid".format(self.config.log_dir))
        # TODO: Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # TODO: Save file for the current model. We want to use the log directory for
        # this. We will name the save file prefix to be "model" inside
        # `self.config.log_dir`
        self.save_file_cur = "{}/model".format(self.config.log_dir)
        # TODO: Save file for the best model. We want to do the same as above,
        # but at the `self.config.save_dir` directory
        self.save_file_best = "{}/model".format(self.config.save_dir)

    def train(self, x_tr, y_tr, x_va, y_va):
        """Training function.

        Parameters
        ----------
        x_tr : ndarray
            Training data.

        y_tr : ndarray
            Training labels.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """

        # ----------------------------------------
        # Preprocess data

        # Report data statistic
        print("Training data before: mean {}, std {}, min {}, max {}".format(
            x_tr.mean(), x_tr.std(), x_tr.min(), x_tr.max()
        ))

        # Normalize data using the normalize function. Note that we are
        # remembering the mean and the range of training data and applying that
        # to the validation/test data later on. We will only compute mean and
        # range to use later. This will be used "inside" the computation graph.
        _, x_tr_mean, x_tr_range = normalize(x_tr)

        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:
            print("Initializing...")
            # TODO: Initialize all variables in the computation graph
            init = tf.global_variables_initializer()
            sess.run(init)
            # TODO: Assign normalization variables from statistics of the train
            # data. Do `sess.run` on the `self.n_assign_op` in a proper way.
            sess.run(
                self.n_assign_op,
                feed_dict={
                    self.n_mean_in: x_tr_mean,
                    self.n_range_in: x_tr_range,
                }
            )

            # TODO: Test on validation data to record initial
            # performance. Again, do `sess.run` but fetch the `self.summary_op`
            # and also the `self.global_step` to write to be used when writing
            # the summary function. For the `feed_dict` you probably want to
            # feed the validation data and labels.
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

            # TODO: Write validation Summary. Use `add_summary` on the validation
            # summary writer with the results you fetched above.
            self.summary_va.add_summary(res["summary"], global_step=res["global_step"])

            print("Training...")
            batch_size = config.batch_size
            num_epoch = config.num_epoch
            num_batch = len(x_tr) // batch_size
            best_acc = 0
            # For each epoch. Note the fancy `trange`!
            for idx_epoch in trange(num_epoch):
                # Create a random order to go through the data
                ind_data = np.random.permutation(len(x_tr))
                # For each training batch
                for idx_batch in range(num_batch):
                    # Construct batch
                    ind_cur = ind_data[
                        batch_size * idx_batch:batch_size * (idx_batch + 1)
                    ]
                    # I noticed that a lot of you guys did a way better job at
                    # this than me. However, I'm doing it this way because in
                    # some cases x_tr[_i] could be your hdf5 file directly! In
                    # which case, you do not need to load the entire data into
                    # memory :-). This way, you only load them into memory at
                    # this precise moment. Just FYI :-)
                    x_b = np.array([x_tr[_i] for _i in ind_cur])
                    y_b = np.array([y_tr[_i] for _i in ind_cur])
                    # TODO: Optimize, get summary for losses and accuracy, get
                    # global_step. So you want to now fetch `self.optim`,
                    # `self.summary_op`, and `self.global_step`, asll with x_b
                    # and y_b in the feed_dict.
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
                    # TODO: Write Training Summary. Same as above, but using
                    # the training summary writer
                    self.summary_tr.add_summary(res["summary"], global_step=res["global_step"])

                # Write immediate after one epoch. Otherwise, summary writer
                # won't write until he thinks it's time to do so. You can alter
                # this behaviour in another way, but I just wanted to show this
                # to you.
                self.summary_tr.flush()

                # TODO: Test on validation data and report results. Here. we
                # want to fetch not only the `self.summary_op` and
                # `self.global_step` as above, but also the `self.acc`, as we
                # are going to check if this is the best model that we've
                # trained so far.
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
                # TODO: Write Validation Summary. Same as a bit above
                self.summary_va.add_summary(res["summary"], global_step=res["global_step"])
                # Write immediate for validation
                self.summary_va.flush()
                # TODO: Save current model to resume later if we want to. Note
                # that we will never do this for our assignment, but hey, why
                # not. Use the `save` method for the `self.saver_cur`. Be sure
                # to say `write_meta_graph=False` or otherwise you will end up
                # with a VERY big save file. Also, you want to pass
                # `self.global_step` directly to the saver instance, instead of
                # the fetched value, as TF wants it that way for some reason.
                self.saver_cur.save(sess, self.save_file_cur, global_step=self.global_step, write_meta_graph=False)

                # TODO: If best validation accuracy, update W_best, b_best, and
                # best accuracy. We will only return the best W and b
                if res["accuracy"] > best_acc:
                    best_acc = res["accuracy"]

                    # TODO: Save the best model. Similar to above save, but
                    # this time, we will simply save using the
                    # `self.saver_best` saver instance and at
                    # `self.save_file_best`. We will also not pass the
                    # `self.global_step` as we only want a single save
                    # file. Again, let's not save the meta graph.
                    self.saver_best.save(sess, self.save_file_best, write_meta_graph=False)

    def test(self, x_te, y_te):
        """Test routine"""

        with tf.Session() as sess:
            # TODO: Load the best model. I already went through this at Lecture
            # 11. Look for the `latest_checkpoint` at the
            # `self.config.save_dir`. And then run `self.saver_best.restore` in
            # an appropriate way.
            latest_checkpoint = tf.train.latest_checkpoint(self.config.save_dir)
            if latest_checkpoint is not None:
                print("restored")
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                    )
            else:
                print("restore failed")
            # TODO: Test on the test data. Simply fetch the accuracy so that we
            # can print it.

            res = sess.run(
                    fetches = {
                        "accuracy": self.acc,
                    },
                    feed_dict={
                        self.x_in: x_te,
                        self.y_in: y_te,
                    }
            )

            # Report (print) test result
            print("Test accuracy with the best model is {}".format(
                res["accuracy"]))


def main(config):
    """The main function."""

    # ----------------------------------------
    # Load cifar10 train data
    print("Reading training data...")
    data_trva, y_trva = load_data(config.data_dir, "train")

    # ----------------------------------------
    # Load cifar10 test data
    print("Reading test data...")
    data_te, y_te = load_data(config.data_dir, "test")

    # ----------------------------------------
    # Extract features
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

    # Randomly shuffle data and labels. IMPORANT: make sure the data and label
    # is shuffled with the same random indices so that they don't get mixed up!
    idx_shuffle = np.random.permutation(len(x_trva))
    x_trva = x_trva[idx_shuffle]
    y_trva = y_trva[idx_shuffle]

    # Change type to float32 and int64 since we are going to use that for
    # TensorFlow.
    x_trva = x_trva.astype("float32")
    y_trva = y_trva.astype("int64")

    # ----------------------------------------
    # Simply select the last 20% of the training data as validation dataset.
    num_tr = int(len(x_trva) * 0.8)

    x_tr = x_trva[:num_tr]
    x_va = x_trva[num_tr:]
    y_tr = y_trva[:num_tr]
    y_va = y_trva[num_tr:]

    # ----------------------------------------
    # Init network class
    mynet = MyNetwork(x_tr.shape, config)

    # ----------------------------------------
    # Train
    # Run training
    mynet.train(x_tr, y_tr, x_va, y_va)

    # ----------------------------------------
    # Test
    mynet.test(x_te, y_te)


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


#
# solution.py ends here
