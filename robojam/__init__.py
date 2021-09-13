"""RoboJam MDRNN Model. The new Keras version!
Charles P. Martin, 2018
University of Oslo, Norway.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import mdn
from .sample_data import *


def build_robojam_model(seq_len=30, hidden_units=256, num_mixtures=5, layers=2, time_dist=True, inference=False, compile_model=True, print_summary=True, predict_moving=False):
    """Builds a RoboJam MDRNN model for training or inference.

    Keyword Arguments:
    seq_len : sequence length to unroll
    hidden_units : number of LSTM units in each layer
    num_mixtures : number of mixture components (5-10 is good)
    layers : number of layers (2 is good)
    time_dist : time distributed or not (default True)
    inference : inference network or training (default False)
    compile_model : compiles the model (default True)
    print_summary : print summary after creating mdoe (default True)
    """
    print("Building RoboJam Model...")
    if predict_moving:
        out_dim = 4  # x, y, dt, m
    else:
        out_dim = 3  # x, y, dt
    # Set up training mode
    stateful = False
    batch_shape = None
    # Set up inference mode.
    if inference:
        stateful = True
        batch_shape = (1, 1, out_dim)
    inputs = tf.keras.Input(name='inputs', batch_shape=batch_shape) # shape=(seq_len,out_dim)
    lstm_in = inputs  # starter input for lstm
    for layer_i in range(layers):
        ret_seq = True
        if (layer_i == layers - 1) and not time_dist:
            # return sequences false if last layer, and not time distributed.
            ret_seq = False
        lstm_out = tf.keras.layers.LSTM(hidden_units, name='lstm'+str(layer_i), return_sequences=ret_seq, stateful=stateful)(lstm_in)
        lstm_in = lstm_out

    mdn_layer = mdn.MDN(out_dim, num_mixtures, name='mdn_outputs')
    if time_dist:
        mdn_layer = tf.keras.layers.TimeDistributed(mdn_layer, name='td_mdn')
    mdn_out = mdn_layer(lstm_out)  # apply mdn
    model = tf.keras.Model(inputs=inputs, outputs=mdn_out)

    if compile_model:
        loss_func = mdn.get_mixture_loss_func(out_dim, num_mixtures)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(loss=loss_func, optimizer=optimizer)

    model.summary()
    return model


def load_robojam_inference_model(model_file="", layers=2, units=512, mixtures=5, predict_moving=False):
    """Returns a RoboJam model loaded from a file"""
    # TODO: make this parse the name to get the hyperparameters.
    # Decoding Model
    decoder = decoder = build_robojam_model(seq_len=1, hidden_units=units, num_mixtures=mixtures, layers=layers, time_dist=False, inference=True, compile_model=False, print_summary=True, predict_moving=predict_moving)
    decoder.load_weights(model_file)
    return decoder


# Performance Helper Functions
SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.


def perf_df_to_array(perf_df, include_moving=False):
    """Converts a dataframe of a performance into array a,b,dt format."""
    perf_df['dt'] = perf_df.time.diff()
    perf_df.dt = perf_df.dt.fillna(0.0)
    # Clean performance data
    # Tiny Performance bounds defined to be in [[0,1],[0,1]], edit to fix this.
    perf_df.at[perf_df[perf_df.dt > 5].index, 'dt'] = 5.0
    perf_df.at[perf_df[perf_df.dt < 0].index, 'dt'] = 0.0
    perf_df.at[perf_df[perf_df.x > 1].index, 'x'] = 1.0
    perf_df.at[perf_df[perf_df.x < 0].index, 'x'] = 0.0
    perf_df.at[perf_df[perf_df.y > 1].index, 'y'] = 1.0
    perf_df.at[perf_df[perf_df.y < 0].index, 'y'] = 0.0
    if include_moving:
        output = np.array(perf_df[['x', 'y', 'dt', 'moving']])
    else:
        output = np.array(perf_df[['x', 'y', 'dt']])
    return output


def perf_array_to_df(perf_array):
    """Converts an array of a performance (a,b,dt(,moving) format) into a dataframe."""
    perf_array = perf_array.T
    perf_df = pd.DataFrame({'x': perf_array[0], 'y': perf_array[1], 'dt': perf_array[2]})
    if len(perf_array) == 4:
        perf_df['moving'] = perf_array[3]
    else:
        # As a rule of thumb, could classify taps with dt>0.1 as taps, dt<0.1 as moving touches.
        perf_df['moving'] = 1
        perf_df.at[perf_df[perf_df.dt > 0.1].index, 'moving'] = 0
    perf_df['time'] = perf_df.dt.cumsum()
    perf_df['z'] = 38.0
    perf_df = perf_df.set_index(['time'])
    return perf_df[['x', 'y', 'z', 'moving']]


def random_touch(with_moving=False):
    """Generate a random tiny performance touch."""
    if with_moving:
        return np.array([np.random.rand(), np.random.rand(), 0.01, 0])
    else:
        return np.array([np.random.rand(), np.random.rand(), 0.01])


def constrain_touch(touch, with_moving=False):
    """Constrain touch values from the MDRNN"""
    touch[0] = min(max(touch[0], 0.0), 1.0)  # x in [0,1]
    touch[1] = min(max(touch[1], 0.0), 1.0)  # y in [0,1]
    touch[2] = max(touch[2], 0.001)  # dt # define minimum time step
    if with_moving:
        touch[3] = np.greater(touch[3], 0.5) * 1.0
    return touch


def generate_random_tiny_performance(model, n_mixtures, first_touch, time_limit=5.0, steps_limit=1000, temp=1.0, sigma_temp=0.0, predict_moving=False):
    """Generates a tiny performance up to 5 seconds in length."""
    if predict_moving:
        out_dim = 4
    else:
        out_dim = 3
    time = 0
    steps = 0
    previous_touch = first_touch
    performance = [previous_touch.reshape((out_dim,))]
    while (steps < steps_limit and time < time_limit):
        params = model.predict(previous_touch.reshape(1,1,out_dim) * SCALE_FACTOR)
        previous_touch = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
        output_touch = previous_touch.reshape(out_dim,)
        output_touch = constrain_touch(output_touch, with_moving=predict_moving)
        performance.append(output_touch.reshape((out_dim,)))
        steps += 1
        time += output_touch[2]
    return np.array(performance)


def condition_and_generate(model, perf, n_mixtures, time_limit=5.0, steps_limit=1000, temp=1.0, sigma_temp=0.0, predict_moving=False):
    """Conditions the network on an existing tiny performance, then generates a new one."""
    if predict_moving:
        out_dim = 4
    else:
        out_dim = 3
    time = 0
    steps = 0
    # condition
    for touch in perf:
        params = model.predict(touch.reshape(1, 1, out_dim) * SCALE_FACTOR)
        previous_touch = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
        output = [previous_touch.reshape((out_dim,))]
    # generate
    while (steps < steps_limit and time < time_limit):
        params = model.predict(previous_touch.reshape(1, 1, out_dim) * SCALE_FACTOR)
        previous_touch = mdn.sample_from_output(params[0], out_dim, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
        output_touch = previous_touch.reshape(out_dim,)
        output_touch = constrain_touch(output_touch, with_moving=predict_moving)
        output.append(output_touch.reshape((out_dim,)))
        steps += 1
        time += output_touch[2]
    net_output = np.array(output)
    return net_output


def divide_performance_into_swipes(perf_df):
    """Divides a performance into a sequence of swipe dataframes for plotting."""
    touch_starts = perf_df[perf_df.moving == 0].index
    performance_swipes = []
    remainder = perf_df
    for att in touch_starts:
        swipe = remainder.iloc[remainder.index < att]
        performance_swipes.append(swipe)
        remainder = remainder.iloc[remainder.index >= att]
    performance_swipes.append(remainder)
    return performance_swipes
