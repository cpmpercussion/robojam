#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import time
from io import StringIO
import pandas as pd
import numpy as np
import mdn
import keras
from keras.layers import Dense, Input

from flask import Flask, request
from flask_cors import CORS


SCALE_FACTOR = 10  # scales input and output from the model. Should be the same between training and inference.


def build_robojam_model(seq_len=30, hidden_units=256, num_mixtures=5, layers=2, time_dist=True, inference=False, compile_model=True, print_summary=True):
    """Builds a RoboJam MDRNN model for training or inference.
    
    Keyword Arguments:
    seq_len : 
    hidden_units : 
    num_mixtures : 
    layers : 
    time_dist : 
    inference : 
    compile_model : 
    print_summary : 
    """
    print("Building RoboJam Model...")
    out_dim = 3 # fixed in the model. x, t, dt, could add extra dim for moving/not moving
    # Set up training mode
    stateful = False
    batch_shape = None
    # Set up inference mode.
    if inference:
        stateful = True
        batch_shape = (1,1,out_dim)
    inputs = keras.layers.Input(shape=(seq_len,out_dim), name='inputs', batch_shape=batch_shape)
    lstm_in = inputs # starter input for lstm
    for layer_i in range(layers):
        ret_seq = True
        if (layer_i == layers - 1) and not time_dist:
            # return sequences false if last layer, and not time distributed.
            ret_seq = False
        lstm_out = keras.layers.LSTM(hidden_units, name='lstm'+str(layer_i), return_sequences=ret_seq, stateful=stateful)(lstm_in)
        lstm_in = lstm_out

    mdn_layer = mdn.MDN(out_dim, num_mixtures, name='mdn_outputs')
    if time_dist:
        mdn_layer = keras.layers.TimeDistributed(mdn_layer, name='td_mdn')
    mdn_out = mdn_layer(lstm_out)  # apply mdn
    model = keras.models.Model(inputs=inputs, outputs=mdn_out)

    if compile_model:
        loss_func = mdn.get_mixture_loss_func(out_dim,num_mixtures)
        optimizer = keras.optimizers.Adam() # keras.optimizers.Adam(lr=0.0001))
        model.compile(loss=loss_func, optimizer=optimizer)

    model.summary()
    return model


# Performance Helper Functions


def perf_df_to_array(perf_df):
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
    return np.array(perf_df[['x', 'y', 'dt']])


def perf_array_to_df(perf_array):
    """Converts an array of a performance (a,b,dt format) into a dataframe."""
    perf_array = perf_array.T
    perf_df = pd.DataFrame({'x': perf_array[0], 'y': perf_array[1], 'dt': perf_array[2]})
    perf_df['time'] = perf_df.dt.cumsum()
    perf_df['z'] = 38.0
    # As a rule of thumb, could classify taps with dt>0.1 as taps, dt<0.1 as moving touches.
    perf_df['moving'] = 1
    perf_df.at[perf_df[perf_df.dt > 0.1].index, 'moving'] = 0
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

def generate_random_tiny_performance(model, n_mixtures, first_touch, time_limit=5.0, steps_limit=1000, temp=1.0, sigma_temp=0.0):
    """Generates a tiny performance up to 5 seconds in length."""
    time = 0
    steps = 0
    previous_touch = first_touch
    performance = [previous_touch.reshape((3,))]
    while (steps < steps_limit and time < time_limit):
        params = model.predict(previous_touch.reshape(1,1,3) * SCALE_FACTOR)
        previous_touch = mdn.sample_from_output(params[0], 3, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
        output_touch = previous_touch.reshape(3,)
        output_touch = constrain_touch(output_touch)
        performance.append(output_touch.reshape((3,)))
        steps += 1
        time += output_touch[2]
    return np.array(performance)


def condition_and_generate(model, perf, n_mixtures, time_limit=5.0, steps_limit=1000, temp=1.0, sigma_temp=0.0):
    """Conditions the network on an existing tiny performance, then generates a new one."""
    time = 0
    steps = 0
    # condition
    for touch in perf:
        params = model.predict(touch.reshape(1,1,3) * SCALE_FACTOR)
        previous_touch = mdn.sample_from_output(params[0], 3, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
        output = [previous_touch.reshape((3,))]
    # generate
    while (steps < steps_limit and time < time_limit):
        params = model.predict(previous_touch.reshape(1,1,3) * SCALE_FACTOR)
        previous_touch = mdn.sample_from_output(params[0], 3, n_mixtures, temp=temp, sigma_temp=sigma_temp) / SCALE_FACTOR
        output_touch = previous_touch.reshape(3,)
        output_touch = constrain_touch(output_touch)
        output.append(output_touch.reshape((3,)))
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

def plot_2D(perf_df, name="foo", saving=False):
    """Plot a 2D representation of a performance 2D"""
    swipes = divide_performance_into_swipes(perf_df)
    plt.figure(figsize=(8, 8))
    for swipe in swipes:
        p = plt.plot(swipe.x, swipe.y, 'o-')
        plt.setp(p, color=gen_colour, linewidth=5.0)
    plt.ylim(1.0,0)
    plt.xlim(0,1.0)
    plt.xticks([])
    plt.yticks([])
    if saving:
        plt.savefig(name+".png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def plot_double_2d(perf1, perf2, name="foo", saving=False):
    """Plot two performances in 2D"""
    plt.figure(figsize=(8, 8))
    swipes = divide_performance_into_swipes(perf1)
    for swipe in swipes:
        p = plt.plot(swipe.x, swipe.y, 'o-')
        plt.setp(p, color=input_colour, linewidth=5.0)
    swipes = divide_performance_into_swipes(perf2)
    for swipe in swipes:
        p = plt.plot(swipe.x, swipe.y, 'o-')
        plt.setp(p, color=gen_colour, linewidth=5.0)
    plt.ylim(1.0,0)
    plt.xlim(0,1.0)
    plt.xticks([])
    plt.yticks([])
    if saving:
        plt.savefig(name+".png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def load_robojam_model(model_file=""):
    # Decoding Model
    decoder = decoder = build_robojam_model(seq_len=1, hidden_units=512, num_mixtures=5, layers=2, time_dist=False, inference=True, compile_model=False, print_summary=True)
    decoder.load_weights(model_file)
    return decoder

# Start server.


app = Flask(__name__)
cors = CORS(app)


# Network hyper-parameters:
N_MIX = 5
N_LAYERS = 2
N_UNITS = 256

TEMP = 1.5
SIG_TEMP = 0.01
MODEL_FILE = 'models/robojam-td-model-E12-VL-4.57.hdf5'


@app.route("/api/predict", methods=['POST'])
def reaction():
    """Produces a Reaction Performance using the MDRNN."""
    print("Responding to a prediction request.")
    start = time.time()
    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        input_perf = json.loads(params['perf'])
    else:
        print("Perf in data as string.")
        params = json.loads(data)
        input_perf = params['perf']
    input_perf_df = pd.read_csv(StringIO(input_perf), parse_dates=False)
    #print(input_perf_df)
    input_perf_array = perf_df_to_array(input_perf_df)
    # Run the response prediction:
    # condition_and_generate(model, perf, n_mixtures, time_limit=5.0, steps_limit=1000, temp=1.0):
    net.reset_states() # reset LSTM state.
    out_perf = condition_and_generate(net, input_perf_array, N_MIX, temp=TEMP, sigma_temp=SIG_TEMP)
    out_df = perf_array_to_df(out_perf)
    out_df.set_value(out_df[:1].index, 'moving', 0)  # set first touch to a tap
    out_perf_string = out_df.to_csv()
    json_data = json.dumps({'response': out_perf_string})
    print("Completed request, time was: %f" % (time.time() - start))
    return json_data


if __name__ == "__main__":
    """Start a TinyPerformance MDRNN Server"""
    print("Starting TinyPerformance Responder.")
    print('Loading the model')
    net = load_robojam_model(model_file=MODEL_FILE)
    print('Starting the API')
    app.run(host='0.0.0.0', ssl_context=('keys/cert.pem', 'keys/key.pem'))


# Command line tests.
# curl -i -k -X POST -H "Content-Type:application/json" https://127.0.0.1:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.005213, 0.711230, 0.070856, 25.524292, 0\n0.097298, 0.719251, 0.062834, 25.524292, 1\n0.126225, 0.719251, 0.057487, 25.524292, 1\n0.194616, 0.707219, 0.045455, 38.290771, 1\n0.212923, 0.704545, 0.045455, 38.290771, 1\n0.343579, 0.703209, 0.108289, 38.290771, 1\n0.495085, 0.701872, 0.070856, 38.290771, 1\n0.523921, 0.693850, 0.061497, 38.290771, 1\n0.712066, 0.711230, 0.155080, 38.290771, 1\n0.730294, 0.717914, 0.155080, 38.290771, 1\n0.896367, 0.696524, 0.041444, 38.290771, 1\n1.083786, 0.696524, 0.151070, 38.290771, 1\n1.301470, 0.684492, 0.049465, 38.290771, 1\n1.328134, 0.680481, 0.053476, 38.290771, 1\n1.504139, 0.705882, 0.136364, 38.290771, 1\n1.527875, 0.712567, 0.120321, 38.290771, 1\n1.702672, 0.675134, 0.076203, 38.290771, 1\n1.719294, 0.675134, 0.096257, 38.290771, 1\n1.901434, 0.715241, 0.145722, 38.290771, 1\n1.922717, 0.717914, 0.136364, 38.290771, 1\n2.062994, 0.684492, 0.109626, 38.290771, 1\n2.091680, 0.680481, 0.129679, 38.290771, 1\n2.231362, 0.697861, 0.207219, 38.290771, 1\n2.393213, 0.712567, 0.124332, 38.290771, 1\n2.525774, 0.632353, 0.149733, 38.290771, 1\n2.546701, 0.625668, 0.169786, 38.290771, 1\n2.686487, 0.585561, 0.360963, 38.290771, 1\n2.715316, 0.580214, 0.387701, 38.290771, 1\n2.867526, 0.490642, 0.633690, 38.290771, 1\n2.880361, 0.481283, 0.645722, 38.290771, 1\n3.054443, 0.319519, 0.689840, 38.290771, 1\n3.218741, 0.121658, 0.585561, 38.290771, 1\n3.230362, 0.102941, 0.557487, 38.290771, 1\n3.391456, 0.089572, 0.534759, 38.290771, 1"}'
# curl -i -k -X POST -H "Content-Type:application/json" https://138.197.179.234:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.005213, 0.711230, 0.070856, 25.524292, 0\n0.097298, 0.719251, 0.062834, 25.524292, 1\n0.126225, 0.719251, 0.057487, 25.524292, 1\n0.194616, 0.707219, 0.045455, 38.290771, 1\n0.212923, 0.704545, 0.045455, 38.290771, 1\n0.343579, 0.703209, 0.108289, 38.290771, 1\n0.495085, 0.701872, 0.070856, 38.290771, 1\n0.523921, 0.693850, 0.061497, 38.290771, 1\n0.712066, 0.711230, 0.155080, 38.290771, 1\n0.730294, 0.717914, 0.155080, 38.290771, 1\n0.896367, 0.696524, 0.041444, 38.290771, 1\n1.083786, 0.696524, 0.151070, 38.290771, 1\n1.301470, 0.684492, 0.049465, 38.290771, 1\n1.328134, 0.680481, 0.053476, 38.290771, 1\n1.504139, 0.705882, 0.136364, 38.290771, 1\n1.527875, 0.712567, 0.120321, 38.290771, 1\n1.702672, 0.675134, 0.076203, 38.290771, 1\n1.719294, 0.675134, 0.096257, 38.290771, 1\n1.901434, 0.715241, 0.145722, 38.290771, 1\n1.922717, 0.717914, 0.136364, 38.290771, 1\n2.062994, 0.684492, 0.109626, 38.290771, 1\n2.091680, 0.680481, 0.129679, 38.290771, 1\n2.231362, 0.697861, 0.207219, 38.290771, 1\n2.393213, 0.712567, 0.124332, 38.290771, 1\n2.525774, 0.632353, 0.149733, 38.290771, 1\n2.546701, 0.625668, 0.169786, 38.290771, 1\n2.686487, 0.585561, 0.360963, 38.290771, 1\n2.715316, 0.580214, 0.387701, 38.290771, 1\n2.867526, 0.490642, 0.633690, 38.290771, 1\n2.880361, 0.481283, 0.645722, 38.290771, 1\n3.054443, 0.319519, 0.689840, 38.290771, 1\n3.218741, 0.121658, 0.585561, 38.290771, 1\n3.230362, 0.102941, 0.557487, 38.290771, 1\n3.391456, 0.089572, 0.534759, 38.290771, 1"}'
# curl -i -k -X POST -H "Content-Type:application/json" https://138.197.179.234:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.002468, 0.106414, 0.122449, 20.000000, 0\n0.020841, 0.106414, 0.125364, 20.000000, 1\n0.043218, 0.107872, 0.137026, 20.000000, 1\n0.065484, 0.107872, 0.176385, 20.000000, 1\n0.090776, 0.107872, 0.231778, 20.000000, 1\n0.110590, 0.109329, 0.301749, 20.000000, 1\n0.133338, 0.115160, 0.357143, 20.000000, 1\n0.155677, 0.125364, 0.412536, 20.000000, 1\n0.178238, 0.134111, 0.432945, 20.000000, 1\n0.516467, 0.275510, 0.180758, 20.000000, 0\n0.542726, 0.274052, 0.205539, 20.000000, 1\n0.560772, 0.274052, 0.249271, 20.000000, 1\n0.583259, 0.282799, 0.316327, 20.000000, 1\n0.605750, 0.295918, 0.376093, 20.000000, 1\n0.628259, 0.309038, 0.415452, 20.000000, 1\n0.653835, 0.316327, 0.432945, 20.000000, 1\n0.673523, 0.325073, 0.440233, 20.000000, 1\n1.000294, 0.590379, 0.179300, 20.000000, 0\n1.022137, 0.593294, 0.183673, 20.000000, 1\n1.044706, 0.594752, 0.208455, 20.000000, 1\n1.067020, 0.606414, 0.279883, 20.000000, 1\n1.091137, 0.626822, 0.355685, 20.000000, 1\n1.111968, 0.647230, 0.425656, 20.000000, 1\n1.134535, 0.655977, 0.462099, 20.000000, 1\n1.156987, 0.657434, 0.485423, 20.000000, 1\n1.619212, 0.857143, 0.263848, 20.000000, 0\n1.642492, 0.854227, 0.281341, 20.000000, 1\n1.663123, 0.851312, 0.320700, 20.000000, 1\n1.685776, 0.846939, 0.413994, 20.000000, 1\n1.708192, 0.846939, 0.510204, 20.000000, 1\n1.730717, 0.858601, 0.591837, 20.000000, 1\n1.753953, 0.868805, 0.632653, 20.000000, 1\n1.775862, 0.876093, 0.660350, 20.000000, 1\n4.376275, 0.542274, 0.860058, 20.000000, 0\n4.419554, 0.543732, 0.860058, 20.000000, 1"}'
# curl -i -k -X POST -H "Content-Type:application/json" https://0.0.0.0:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.002468, 0.106414, 0.122449, 20.000000, 0\n0.020841, 0.106414, 0.125364, 20.000000, 1\n0.043218, 0.107872, 0.137026, 20.000000, 1\n0.065484, 0.107872, 0.176385, 20.000000, 1\n0.090776, 0.107872, 0.231778, 20.000000, 1\n0.110590, 0.109329, 0.301749, 20.000000, 1\n0.133338, 0.115160, 0.357143, 20.000000, 1\n0.155677, 0.125364, 0.412536, 20.000000, 1\n0.178238, 0.134111, 0.432945, 20.000000, 1\n0.516467, 0.275510, 0.180758, 20.000000, 0\n0.542726, 0.274052, 0.205539, 20.000000, 1\n0.560772, 0.274052, 0.249271, 20.000000, 1\n0.583259, 0.282799, 0.316327, 20.000000, 1\n0.605750, 0.295918, 0.376093, 20.000000, 1\n0.628259, 0.309038, 0.415452, 20.000000, 1\n0.653835, 0.316327, 0.432945, 20.000000, 1\n0.673523, 0.325073, 0.440233, 20.000000, 1\n1.000294, 0.590379, 0.179300, 20.000000, 0\n1.022137, 0.593294, 0.183673, 20.000000, 1\n1.044706, 0.594752, 0.208455, 20.000000, 1\n1.067020, 0.606414, 0.279883, 20.000000, 1\n1.091137, 0.626822, 0.355685, 20.000000, 1\n1.111968, 0.647230, 0.425656, 20.000000, 1\n1.134535, 0.655977, 0.462099, 20.000000, 1\n1.156987, 0.657434, 0.485423, 20.000000, 1\n1.619212, 0.857143, 0.263848, 20.000000, 0\n1.642492, 0.854227, 0.281341, 20.000000, 1\n1.663123, 0.851312, 0.320700, 20.000000, 1\n1.685776, 0.846939, 0.413994, 20.000000, 1\n1.708192, 0.846939, 0.510204, 20.000000, 1\n1.730717, 0.858601, 0.591837, 20.000000, 1\n1.753953, 0.868805, 0.632653, 20.000000, 1\n1.775862, 0.876093, 0.660350, 20.000000, 1\n4.376275, 0.542274, 0.860058, 20.000000, 0\n4.419554, 0.543732, 0.860058, 20.000000, 1"}'
