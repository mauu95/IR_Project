from utility import *
from dictionary import *
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
import rnn
import rnn_att
import rnn_multi_att
import transformer
import math
import torch.nn as nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch

def smooth(arr, window_size):
    smoothed = []
    for i in range(len(arr) - window_size + 1):
        window_average = round(np.sum(arr[i:i+window_size]) / window_size, 2)
        smoothed.append(window_average)
    return smoothed

def avg(arr):
    return round(np.sum(arr) / len(arr), 3)

def avgd(arr):
    n = len(arr)
    halfn = math.floor(n/2)
    arr1 = arr[:halfn]
    arr2 = arr[halfn:]
    diff = avg(arr1) - avg(arr2)
    return round(diff, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def reset(model):
    model.encoder = model.Encoder(model.dictionary.n_words).to(device)
    model.decoder = model.Decoder(model.dictionary.n_words).to(device)
    model.enc_optimizer = optim.SGD(model.encoder.parameters(), lr=learning_rate)
    model.dec_optimizer = optim.SGD(model.decoder.parameters(), lr=learning_rate)
    model.criterion = nn.NLLLoss()
    model.enc_scheduler = lr_scheduler.ReduceLROnPlateau(model.enc_optimizer, patience=patience, factor=scheduler_factor)
    model.dec_scheduler = lr_scheduler.ReduceLROnPlateau(model.dec_optimizer, patience=patience, factor=scheduler_factor)
    

smooth_degree = 20
models = [rnn, rnn_att, rnn_multi_att, transformer]
x_values = [i for i in range(n_iteration - smooth_degree + 1)]
n_experiment = 10

for model in models:
    losses = [0] * (n_iteration - smooth_degree + 1)
    for i in range(n_experiment):
        print(f'Start training {model.name} {i+1} of {n_experiment}')
        reset(model)
        losses = np.add(losses, smooth(model.train_iters(), smooth_degree))
    losses = np.divide(losses, n_experiment)
    plt.plot(x_values, losses, label=f'{model.name} (AVG={avg(losses)}, AVGD={avgd(losses)})')

plt.title(f'Average of {n_experiment} experiments. LR(scheduler={use_scheduler}):{learning_rate}, n_iterations:{n_iteration}, n_samples:{n_training_samples}')
plt.legend()
plt.show()

