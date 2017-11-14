import sample_data
import tiny_performance_loader
import mixture_rnn

# experiment 1

# # Load Data
# data_loader = tiny_performance_loader.TinyPerformanceLoader()
# tiny_performance_corpus = data_loader.single_sequence_corpus()
# loader = sample_data.SequenceDataLoader(num_steps=121, batch_size=64, corpus=tiny_performance_corpus)
# net = mixture_rnn.MixtureRNN(mode=mixture_rnn.NET_MODE_TRAIN, n_hidden_units=256, n_mixtures=16, batch_size=64, sequence_length=120, n_layers=2)
# # Train
# EPOCHS = 50
# losses = net.train(loader, EPOCHS, saving=True)
# print(losses)

# experiment 2
SEQ_LEN = 64
BATCH_SIZE = 64
HIDDEN_UNITS = 64
LAYERS = 2
MIXES = 8
EPOCHS = 1

# Load Data
data_loader = tiny_performance_loader.TinyPerformanceLoader()
tiny_performance_corpus = data_loader.single_sequence_corpus()
loader = sample_data.SequenceDataLoader(num_steps=SEQ_LEN + 1, batch_size=BATCH_SIZE, corpus=tiny_performance_corpus)
net = mixture_rnn.MixtureRNN(mode=mixture_rnn.NET_MODE_TRAIN, n_hidden_units=HIDDEN_UNITS, n_mixtures=MIXES, batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_layers=LAYERS)
losses = net.train(loader, EPOCHS, saving=True)
print(losses)
