from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from . import mixture_2d_normals
from . import mixture_1d_normals

tf.logging.set_verbosity(tf.logging.INFO)  # set logging.

NET_MODE_TRAIN = 'train'
NET_MODE_RUN = 'run'
MODEL_DIR = "../models/"
LOG_PATH = "./output-logs/"

SUMMARY_WRITING_FREQUENCY = 20  # write the summary variables every 'n' steps. 20 might be good for big models.
VALIDATION_PRINTING_FREQUENCY = 100  # write the summary variables every 'n' steps. 20 might be good for big models.
LOSS_PRINTING_FREQUENCY = 1000  # print the loss to the command line every 'n' steps. 100 for testing, 1-2000 for training.
CHECKPOINT_WRITING_FREQUENCY = 2000  # step frequency to save a checkpoint, only one checkpoint every hour is kept however.
RANDOM_SEED = 2345  # TF Random Seed.


class MixtureRNN(object):
    """Mixture Density Network RNN for generating touchscreen interaction data. Includes a mxture
    of 2D normals for modelling space, and a mixture of 1D normals for modelling time."""

    def __init__(self, mode=NET_MODE_TRAIN, n_hidden_units=128, n_mixtures=24, batch_size=64, sequence_length=128, n_layers=1):
        """
        Initialise the MixtureRNN network.
        Keyword Arguments:
        mode -- Use 'run' for evaluation graph and 'train' for training graph.
        n_hidden_units -- Number of LSTM units in each layer (default=128)
        n_mixtures -- Number of normal distributions in each mixture model (default=24)
        batch_size -- Batch size for training (default=64)
        sequence_length -- Sequence length for RNN unrolling (default=128)
        n_layers -- Number of LSTM layers in network (default=1)
        """
        # hyperparameters
        self.mode = mode
        self.n_hidden_units = n_hidden_units
        self.n_rnn_layers = n_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.st_dev = 0.5
        self.n_mixtures = n_mixtures  # number of mixtures
        self.n_input_units = 3  # Number of dimensions of the input (and sampled output) data
        self.mdn_splits = 9  # (pi, sigma_1, sigma_2, mu_1, mu_2, rho) + (pi_2, sigma_3, mu_3)
        self.n_output_units = n_mixtures * self.mdn_splits  # KMIX * self.mdn_splits
        self.lr = 1e-4  # could be 1e-3
        self.grad_clip = 1.0
        self.state = None
        self.use_input_dropout = False
        if self.mode is NET_MODE_TRAIN:
            self.use_input_dropout = True
        self.dropout_prob = 0.90
        self.run_name = self.get_run_name()

        tf.reset_default_graph()
        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            tf.set_random_seed(RANDOM_SEED)
            with tf.name_scope('input'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, self.n_input_units], name="x")  # input
                self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, self.n_input_units], name="y")  # target
                self.rnn_outputs, self.init_state, self.final_state = self.recurrent_network(self.x)
            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.n_hidden_units], name="reshape_rnn_outputs")

            output_params = self.fully_connected_layer(self.rnn_outputs, self.n_hidden_units, self.n_output_units)
            position_params, time_params = tf.split(output_params, [self.n_mixtures * 6, self.n_mixtures * 3], axis=1, name="time_space_split")

            self.pis, self.scales_1, self.scales_2, self.locs_1, self.locs_2, self.corr = mixture_2d_normals.split_tensor_to_mixture_parameters(position_params)
            self.time_pis, self.time_scales, self.time_locs = mixture_1d_normals.split_tensor_to_mixture_parameters(time_params)
            self.add_parameter_summaries()  # add summaries of the mixture parameters for great good.
            # Saver: keeps last 5 checkpoints as well as one every hour.
            self.saver = tf.train.Saver(name="saver", max_to_keep=100, keep_checkpoint_every_n_hours=1)
            if self.mode is NET_MODE_TRAIN:
                tf.logging.info("Loading Training Operations")
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                with tf.name_scope('labels'):
                    self.y_reshaped = tf.reshape(self.y, [-1, self.n_input_units], name="reshape_labels")
                    [self.y1_data, self.y2_data, self.y3_data] = tf.split(self.y_reshaped, 3, 1)
                # Cost and Accuracy Operations
                self.prob_space = mixture_2d_normals.tf_2d_mixture_prob(self.pis, self.locs_1, self.locs_2, self.scales_1, self.scales_2, self.corr, self.y1_data, self.y2_data)
                self.prob_time = mixture_1d_normals.tf_1d_mixture_prob(self.time_pis, self.time_locs, self.time_scales, self.y3_data)
                self.prob_func = self.prob_space * self.prob_time
                epsilon = 1e-6
                # loss_func = mixture_2d_normals.get_lossfunc(self.pis, self.locs_1, self.locs_2, self.scales_1, self.scales_2, self.corr, self.y1_data, self.y2_data) + mixture_1d_normals.get_lossfunc(self.time_pis, self.time_locs, self.time_scales, self.y3_data)
                loss_func = tf.negative(tf.log(self.prob_space + epsilon) + tf.log(self.prob_time + epsilon))  # avoid log(0)
                self.cost = tf.reduce_mean(loss_func, name="mean_cost")
                self.accuracy = tf.reduce_mean(self.prob_func, name="mean_accuracy")
                # Optimiser and training
                optimizer = tf.train.AdamOptimizer(self.lr)
                gvs = optimizer.compute_gradients(self.cost)
                g = self.grad_clip
                capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
                self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')
                self.training_state = None
                # Logging Training Variables
                tf.summary.scalar("accuracy", self.accuracy)
                tf.summary.scalar("cost_summary", self.cost)
                # tf.summary.histogram("prob_space", self.prob_space)
                # tf.summary.histogram("prob_time", self.prob_time)
                # tf.summary.histogram("accuracy", self.prob_func)

            if self.mode is NET_MODE_RUN:
                tf.logging.info("Loading Running Operations")
                # TODO: write a sketch-RNN version of the sampling function?
            # Summaries
            self.summaries = tf.summary.merge_all()

        if self.mode is NET_MODE_TRAIN:
            self.writer = tf.summary.FileWriter(LOG_PATH + self.run_name + '/', graph=self.graph)
        train_vars_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        tf.logging.info("done initialising: %s vars: %d", self.model_name(), train_vars_count)

    def fully_connected_layer(self, X, in_dim, out_dim):
        with tf.name_scope('rnn_to_mdn'):
            W = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=self.st_dev, dtype=tf.float32, seed=RANDOM_SEED))
            b = tf.Variable(tf.random_normal([1, out_dim], stddev=self.st_dev, dtype=tf.float32, seed=RANDOM_SEED))
            output = tf.matmul(X, W) + b
        return output

    def add_parameter_summaries(self):
        """ Adds summaries of the mixture model parameters for reality checks during training and testing. """
        tf.summary.histogram("pos_pis", self.pis)
        tf.summary.histogram("pos_scales_1", self.scales_1)
        tf.summary.histogram("pos_scales_2", self.scales_2)
        tf.summary.histogram("pos_locs_1", self.locs_1)
        tf.summary.histogram("pos_locs_2", self.locs_2)
        tf.summary.histogram("pos_corr", self.corr)
        tf.summary.histogram("time_pis", self.time_pis)
        tf.summary.histogram("time_scales", self.time_scales)
        tf.summary.histogram("time_locs", self.time_locs)

    def recurrent_network(self, X):
        """ Create the RNN part of the network. """
        with tf.name_scope('recurrent_network'):
            cells_list = [tf.contrib.rnn.LSTMCell(self.n_hidden_units, state_is_tuple=True) for _ in range(self.n_rnn_layers)]
            if self.use_input_dropout:
                cells_list = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_prob) for cell in cells_list]
            cell = tf.contrib.rnn.MultiRNNCell(cells_list, state_is_tuple=True)
            init_state = cell.zero_state(self.batch_size, tf.float32)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(
                cell,
                X,
                initial_state=init_state,
                time_major=False,
                dtype=tf.float32,
                scope='RNN'
            )
        return rnn_outputs, init_state, final_state

    def model_name(self):
        """Returns the name of the present model for saving to disk"""
        return "mdrnn-2d-1d-" + str(self.n_rnn_layers) + "layers-" + str(self.n_hidden_units) + "units-" + str(self.n_mixtures) + "mixtures"

    def get_run_name(self):
        out = self.model_name() + "-"
        out += time.strftime("%Y%m%d-%H%M%S")
        return out

    def train_batch(self, batch, sess):
        """Train the network on one batch"""
        # batch is an array of shape (batch_size, sequence_length + 1, n_input_units)
        batch_x = batch[:, :self.sequence_length, :]
        batch_y = batch[:, 1:, :]
        feed = {self.x: batch_x, self.y: batch_y}
        # should the LSTM state be wiped after each batch?
        # if self.training_state is not None:
        #     feed[self.init_state] = self.training_state
        training_loss_current, self.training_state, _, summary, step = sess.run([self.cost, self.final_state, self.train_op, self.summaries, self.global_step], feed_dict=feed)
        if (step % SUMMARY_WRITING_FREQUENCY == 0):
            #  Write Summaries every SUMMARY_WRITING_FREQUENCY steps. (to save disk space).
            self.writer.add_summary(summary, step)
        if (step % CHECKPOINT_WRITING_FREQUENCY == 0):
            #  Save a model checkpoint every CHECKPOINT_WRITING_FREQUENCY steps.
            checkpoint_path = LOG_PATH + self.run_name + '/' + self.model_name()
            tf.logging.info('saving model %s, global_step %d.', checkpoint_path, step)
            self.saver.save(sess, checkpoint_path, global_step=step)
        return training_loss_current, step

    def train_epoch(self, batches, sess):
        """Train the network on one epoch of training data."""
        total_training_loss = 0
        epoch_steps = 0
        total_steps = len(batches)
        step = 0
        for b in batches:
            training_loss, step = self.train_batch(b, sess)
            if np.isnan(training_loss):
                tf.logging.info('training loss was NaN at step %d', step)
                # raise ValueError('Training loss was NaN for current batch.')
            epoch_steps += 1
            total_training_loss += training_loss
            if (epoch_steps % LOSS_PRINTING_FREQUENCY == 0):
                tf.logging.info("trained batch: %d of %d; loss was %f", epoch_steps, total_steps, training_loss)
        return (total_training_loss / epoch_steps), step

    def train(self, data_manager, num_epochs, saving=True):
        """Train the network for the a number of epochs."""
        self.num_epochs = num_epochs
        tf.logging.info("going to train: %s", self.model_name())
        start_time = time.time()
        training_losses = []
        step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_epochs):
                batches = data_manager.next_epoch()
                # try training, stop if there is a value error (NaN in loss).
                try:
                    epoch_average_loss, step = self.train_epoch(batches, sess)
                except ValueError as err:
                    tf.logging.info("training was aborted due to value error at global_step %d, epoch %d", step, i)
                    tf.logging.info(err.args)
                    break
                training_losses.append(epoch_average_loss)
                tf.logging.info("trained epoch %d of %d", i, self.num_epochs)
            if saving:
                # Save finished model separately.
                tf.logging.info('saving model %s.', self.model_name())
                self.saver.save(sess, MODEL_DIR + self.model_name())
        tf.logging.info("took %d seconds to train.", (time.time() - start_time))
        return training_losses

    def evaluate_batch(self, batch, sess):
        """ Evaluates the network on one batch, i.e., returns loss without running training operations."""
        batch_x = batch[:, :self.sequence_length, :]
        batch_y = batch[:, 1:, :]
        feed = {self.x: batch_x, self.y: batch_y}
        evaluation_loss, evaluation_accuracy, evaluation_state, summary, step = sess.run([self.cost, self.accuracy, self.final_state, self.summaries, self.global_step], feed_dict=feed)
        if (step % SUMMARY_WRITING_FREQUENCY == 0):
            #  Write Summaries every 10 steps. (to save disk space).
            self.writer.add_summary(summary, step)
        return evaluation_loss, evaluation_accuracy, step

    def evaluate(self, data_manager, model_file):
        """ Evaluate the network on each batch in a data_manager."""
        tf.logging.info("going to evaluate: %s", self.model_name())
        start_time = time.time()
        evaluation_losses = []
        evaluation_accuracies = []
        batches = data_manager.next_epoch()
        tf.logging.info("batches have shape: %s", str(batches.shape))
        with tf.Session() as sess:
            self.prepare_model_for_running(sess, model=model_file)
            for ind, b in enumerate(batches):
                eval_loss, eval_acc, step = self.evaluate_batch(b, sess)
                evaluation_losses.append(eval_loss)
                evaluation_accuracies.append(eval_acc)
                tf.logging.info("batch: %d, loss %f, acc %f", ind, eval_loss, eval_acc)
        tf.logging.info("took %d seconds to evaluate.", (time.time() - start_time))
        return evaluation_losses, evaluation_accuracies

    def prepare_model_for_running(self, sess, model=None):
        """Load trained model and reset RNN state."""
        sess.run(tf.global_variables_initializer())
        if model is None:
            self.saver.restore(sess, MODEL_DIR + self.model_name())
        else:
            self.saver.restore(sess, model)
        self.state = None

    def freeze_running_model(self, sess):
        """Freezes and saves the running version of the graph"""
        # note this doesn't work yet!
        if self.mode is not NET_MODE_RUN:
            return
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_node_names = "Accuracy/predictions"
        # Prepare Model fo Running, then save a frozen model.
        with tf.Session() as sess:
            self.prepare_model_for_running(sess)
            self.saver.save(sess, self.model_name() + "frozen")
            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                input_graph_def,  # The graph_def is used to retrieve the nodes
                output_node_names.split(",")  # The output node names are used to select the usefull nodes
            )
            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(self.model_name() + "frozen" + ".pb", "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

    def generate_touch(self, prev_touch, sess, temp=1.0):
        """Generate prediction for a single touch."""
        input_touch = prev_touch.reshape([1, 1, self.n_input_units])  # Give input correct shape for one-at-a-time evaluation.
        if self.state is not None:
            feed = {self.x: input_touch, self.init_state: self.state}
        else:
            feed = {self.x: input_touch}
        pis, locs_1, locs_2, scales_1, scales_2, corr, t_pis, t_locs, t_scales, self.state = sess.run([self.pis, self.locs_1, self.locs_2, self.scales_1, self.scales_2, self.corr, self.time_pis, self.time_locs, self.time_scales, self.final_state], feed_dict=feed)
        x_1, x_2 = mixture_2d_normals.sample_mixture_model(pis[0], locs_1[0], locs_2[0], scales_1[0], scales_2[0], corr[0], temp=temp)
        t = mixture_1d_normals.sample_mixture_model(t_pis[0], t_locs[0], t_scales[0], temp=temp)
        return np.array([x_1, x_2, t])

    def generate_performance(self, first_touch, number, sess, temp=0.5):
        self.prepare_model_for_running(sess)
        previous_touch = first_touch
        performance = [previous_touch.reshape((self.n_input_units,))]
        for i in range(number):
            previous_touch = self.generate_touch(previous_touch, sess, temp=temp)
            performance.append(previous_touch.reshape((self.n_input_units,)))
        return np.array(performance)


def perf_df_to_array(perf_df):
    """Converts a dataframe of a performance into array a,b,dt format."""
    perf_df['dt'] = perf_df.time.diff()
    perf_df.dt = perf_df.dt.fillna(0.0)
    # Clean performance data
    # Tiny Performance bounds defined to be in [[0,1],[0,1]], edit to fix this.
    perf_df.set_value(perf_df[perf_df.dt > 5].index, 'dt', 5.0)
    perf_df.set_value(perf_df[perf_df.dt < 0].index, 'dt', 0.0)
    perf_df.set_value(perf_df[perf_df.x > 1].index, 'x', 1.0)
    perf_df.set_value(perf_df[perf_df.x < 0].index, 'x', 0.0)
    perf_df.set_value(perf_df[perf_df.y > 1].index, 'y', 1.0)
    perf_df.set_value(perf_df[perf_df.y < 0].index, 'y', 0.0)
    return np.array(perf_df[['x', 'y', 'dt']])


def perf_array_to_df(perf_array):
    """Converts an array of a performance (a,b,dt format) into a dataframe."""
    perf_array = perf_array.T
    perf_df = pd.DataFrame({'x': perf_array[0], 'y': perf_array[1], 'dt': perf_array[2]})
    perf_df['time'] = perf_df.dt.cumsum()
    perf_df['z'] = 38.0
    # As a rule of thumb, could classify taps with dt>0.1 as taps, dt<0.1 as moving touches.
    perf_df['moving'] = 1
    perf_df.set_value(perf_df[perf_df.dt > 0.1].index, 'moving', 0)
    perf_df = perf_df.set_index(['time'])
    return perf_df[['x', 'y', 'z', 'moving']]


def random_touch():
    """Generate a random tiny performance touch."""
    return np.array([np.random.rand(), np.random.rand(), 0.01])


def constrain_touch(touch):
    """Constrain touch values from the MDRNN"""
    touch[0] = min(max(touch[0], 0.0), 1.0)  # x in [0,1]
    touch[1] = min(max(touch[1], 0.0), 1.0)  # y in [0,1]
    touch[2] = max(touch[2], 0.001)  # dt # define minimum time step
    return touch


def generate_random_tiny_performance(net, first_touch, time_limit=5.0, steps_limit=1000, temp=1.0, model_file=None):
    """Generates a tiny performance up to 5 seconds in length."""
    time = 0
    steps = 0
    with tf.Session() as sess:
        net.prepare_model_for_running(sess, model=model_file)
        previous_touch = first_touch
        performance = [previous_touch.reshape((3,))]
        while (steps < steps_limit and time < time_limit):
            previous_touch = net.generate_touch(previous_touch, sess, temp=temp)
            output_touch = previous_touch
            output_touch = constrain_touch(output_touch)
            performance.append(output_touch.reshape((3,)))
            steps += 1
            time += output_touch[2]
    return np.array(performance)


def condition_and_generate(net, perf, time_limit=5.0, steps_limit=1000, temp=1.0, model_file=None):
    """Conditions the network on an existing tiny performance, then generates a new one."""
    time = 0
    steps = 0
    with tf.Session() as sess:
        net.prepare_model_for_running(sess, model=model_file)
        # condition
        for touch in perf:
            previous_touch = net.generate_touch(touch, sess, temp=temp)
        output = [previous_touch.reshape((3,))]
        while (steps < steps_limit and time < time_limit):
            previous_touch = net.generate_touch(previous_touch, sess, temp=temp)
            output_touch = previous_touch
            output_touch = constrain_touch(output_touch)
            output.append(output_touch.reshape((3,)))
            steps += 1
            time += output_touch[2]
        net_output = np.array(output)
    return net_output
