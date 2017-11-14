""" N.B.: This model is not working! Suffers from the MDN NaN problem and will not train."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
import mixture_distribution_3d

NET_MODE_TRAIN = 'train'
NET_MODE_RUN = 'run'
MODEL_DIR = "./"
LOG_PATH = "./output-logs/"


class MixtureDensityRNN(object):
    """A Mixture Density RNN for modelling 3-dimensional data."""

    def __init__(self, mode=NET_MODE_TRAIN, n_hidden_units=128, n_mixtures=16, batch_size=32, sequence_length=32, n_layers=1):
        """
        Initialise the MixtureDensityRNN network.
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
        self.st_dev = 1.0
        self.n_mixtures = n_mixtures  # number of mixtures
        self.n_input_units = 3  # Number of dimensions of the input (and sampled output) data
        self.mdn_splits = 7  # (pi, sigma_1, sigma_2, sigma_3, mu_1, mu_2, mu_3)
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
            with tf.name_scope('input'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, self.n_input_units], name="x")  # input
                self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, self.n_input_units], name="y")  # target
                self.rnn_outputs, self.init_state, self.final_state = self.recurrent_network(self.x)
            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.n_hidden_units], name="reshape_rnn_outputs")

            output_params = self.fully_connected_layer(self.rnn_outputs, self.n_hidden_units, self.n_output_units)
            self.pis, self.scales_1, self.scales_2, self.time_scales, self.locs_1, self.locs_2, self.time_locs = mixture_distribution_3d.split_tensor_to_mixture_parameters(output_params)
            input_shape = [self.batch_size, self.n_input_units]  # shape of the output
            self.mixture = mixture_distribution_3d.get_mixture_model(self.pis, self.scales_1, self.scales_2, self.time_scales, self.locs_1, self.locs_2, self.time_locs, input_shape)
            self.sample_op = self.mixture.sample()
            # Saver
            self.saver = tf.train.Saver(name="saver")
            if self.mode is NET_MODE_TRAIN:
                tf.logging.info("Loading Training Operations")
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                with tf.name_scope('labels'):
                    y_reshaped = tf.reshape(self.y, [-1, self.n_input_units], name="reshape_labels")
                    [y1_data, y2_data, y3_data] = tf.split(y_reshaped, 3, 1)
                loss_func = mixture_distribution_3d.get_loss_func(self.mixture, y_reshaped)
                self.cost = tf.reduce_mean(loss_func)
                optimizer = tf.train.AdamOptimizer(self.lr)
                gvs = optimizer.compute_gradients(self.cost)
                g = self.grad_clip
                capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
                self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')
                self.training_state = None
                tf.summary.scalar("cost_summary", self.cost)

            if self.mode is NET_MODE_RUN:
                tf.logging.info("Loading Running Operations")
            # Summaries
            self.summaries = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(LOG_PATH + self.run_name + '/', graph=self.graph)
        train_vars_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        tf.logging.info("done initialising: %s vars: %d", self.model_name(), train_vars_count)

    def fully_connected_layer(self, X, in_dim, out_dim):
        with tf.name_scope('rnn_to_mdn'):
            W = tf.get_variable('output_w', [in_dim, out_dim])
            b = tf.get_variable('output_b', [out_dim])
            output = tf.nn.xw_plus_b(X, W, b)
        tf.summary.histogram("out_weights", W)
        tf.summary.histogram("out_biases", b)
        tf.summary.histogram("out_logits", output)
        return output

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
        return "mdrnn-" + str(self.n_rnn_layers) + "layers-" + str(self.n_hidden_units) + "units-" + str(self.n_mixtures) + "mixtures"

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
        if self.training_state is not None:
            feed[self.init_state] = self.training_state
        training_loss_current, self.training_state, _, summary, step = sess.run([self.cost, self.final_state, self.train_op, self.summaries, self.global_step], feed_dict=feed)
        self.writer.add_summary(summary, step)
        return training_loss_current, step

    def train_epoch(self, batches, sess):
        """Train the network on one epoch of training data."""
        total_training_loss = 0
        epoch_steps = 0
        total_steps = len(batches)
        step = 0
        for b in batches:
            training_loss, step = self.train_batch(b, sess)
            epoch_steps += 1
            total_training_loss += training_loss
            if (epoch_steps % 2000 == 0):
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
                epoch_average_loss, step = self.train_epoch(batches, sess)
                training_losses.append(epoch_average_loss)
                tf.logging.info("trained epoch %d of %d", i, self.num_epochs)
                if saving:
                    checkpoint_path = LOG_PATH + self.run_name + '/' + self.model_name() + ".ckpt"
                    tf.logging.info('saving model %s, global_step %d.', checkpoint_path, step)
                    self.saver.save(sess, checkpoint_path, global_step=step)
            if saving:
                tf.logging.info('saving model %s.', self.model_name())
                self.saver.save(sess, self.model_name())
        tf.logging.info("took %d seconds to train.", (time.time() - start_time))
        return training_losses

    def prepare_model_for_running(self, sess):
        """Load trained model and reset RNN state."""
        sess.run(tf.global_variables_initializer())
        self.saver.restore(sess, MODEL_DIR + self.model_name())
        self.state = None

    def freeze_running_model(self, sess):
        """Freezes and saves the running version of the graph"""
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

    def generate_touch(self, prev_touch, sess):
        """Generate prediction for a single touch."""
        input_touch = prev_touch.reshape([1, 1, self.n_input_units])  # Give input correct shape for one-at-a-time evaluation.
        if self.state is not None:
            feed = {self.x: input_touch, self.init_state: self.state}
        else:
            feed = {self.x: input_touch}
        samp, self.state = sess.run([self.sample_op, self.final_state], feed_dict=feed)
        return samp
        #return np.array([x_1, x_2, t])

    def generate_performance(self, first_touch, number, sess):
        self.prepare_model_for_running(sess)
        previous_touch = first_touch
        performance = [previous_touch.reshape((self.n_input_units,))]
        for i in range(number):
            previous_touch = self.generate_touch(previous_touch, sess)
            performance.append(previous_touch.reshape((self.n_input_units,)))
        return np.array(performance)


#if __name__ == "__main__":
#    train_epochs(30)
