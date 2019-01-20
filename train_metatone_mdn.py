#!/usr/local/bin/python3
import keras
from keras import backend as K
import numpy as np
import tensorflow as tf
import random
import robojam
import os


# Set up environment.
# Only for GPU use:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# Load tiny performance data:
with np.load('./datasets/metatone_dataset.npz') as loaded:
    loaded_raw = loaded['raw_perfs']
    loaded_diff = loaded['diff_perfs']

print("Loaded perfs:", len(loaded_raw), "and", len(loaded_diff))
print("Num touches:", np.sum([len(l) for l in loaded_raw]))

corpus = []

for l in loaded_raw:
    corpus.append(l[:,:-1])

# Model Hyperparameters
SEQ_LEN = 100
SEQ_STEP = 10
HIDDEN_UNITS = 512 #512
N_LAYERS = 2
NUMBER_MIXTURES = 5
TIME_DIST = True

# Training Hyperparameters:
BATCH_SIZE = 64
EPOCHS = 100
VAL_SPLIT = 0.10

# Set random seed for reproducibility
SEED = 2345
random.seed(SEED)
np.random.seed(SEED)

# Restrict corpus to sequences longer than the corpus.
corpus = [l for l in corpus if len(l) > SEQ_LEN+1]
print("Corpus Examples:", len(corpus))

# Prepare training data as X and Y.
slices = []
for seq in corpus:
    slices += robojam.slice_sequence_examples(seq, SEQ_LEN+1, step_size=SEQ_STEP)

X, y = robojam.seq_to_overlapping_format(slices)

X = np.array(X) * robojam.SCALE_FACTOR
y = np.array(y) * robojam.SCALE_FACTOR

print("Number of training examples:")
print("X:", X.shape)
print("y:", y.shape)

# Setup Training Model
model = robojam.build_robojam_model(seq_len=SEQ_LEN, hidden_units=HIDDEN_UNITS, num_mixtures=NUMBER_MIXTURES, layers=2, time_dist=TIME_DIST, inference=False, compile_model=True, print_summary=True)

# Setup callbacks
model_path = "robojam-metatone" + "-layers" + str(N_LAYERS) + "-units" + str(HIDDEN_UNITS) + "-mixtures" + str(NUMBER_MIXTURES) + "-scale" + str(robojam.SCALE_FACTOR)
filepath = model_path + "-E{epoch:02d}-VL{val_loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
terminateOnNaN = keras.callbacks.TerminateOnNaN()
tboard = keras.callbacks.TensorBoard(log_dir='./logs/'+model_path, histogram_freq=2, batch_size=32, write_graph=True, update_freq='epoch')

# Train
history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT, callbacks=[checkpoint, terminateOnNaN, tboard])
#history = model.fit_generator(generator, steps_per_epoch=300, epochs=100, verbose=1, initial_epoch=0)

# Save final Model
model.save('robojam-metatone-final.hdf5')  # creates a HDF5 file of the model

# Plot the loss
# %matplotlib inline
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()
