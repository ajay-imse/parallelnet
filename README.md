# parallelnet
--------------
folder purdue:
Based on arXiv:1903.06379, Enabling spike based backprop in deep NN architectures.

Learning is by back propagation through time, spike data is framed in time steps. For regular MNIST, it is poisson encoded.
Network in  code is for net input - 2x(5c5-2p-5c5-2p-25fc)-10output
with varying decay for each net



folder stbp:
STBP original china group paper
file makeframes.m - reads aedat datas and labels to make frames of 40
msec each, 100 frames for each action in a recording

file ch_frames_stbp.py - reads the .mat frame and labels file and
trains with stbp and tests on whole dataset

file ch_frames_splitnet.py - the split network with static delay for
one half of the network.

file ch_splitnet_kfold - trains and tests with k=4

files confusion mat for full dataset train and test with full and
splitnet
