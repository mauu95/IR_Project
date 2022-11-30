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

smooth_degree = 50

models = [rnn, rnn_att, rnn_multi_att, transformer]

x_values = [i for i in range(n_iteration - smooth_degree + 1)]
for model in models:
    losses = smooth(model.train_iters(), smooth_degree)
    plt.plot(x_values, losses, label=f'{model.name} (AVG={avg(losses)}, AVGD={avgd(losses)})')

plt.title(f'LR(scheduler={use_scheduler}):{learning_rate}, n_iterations:{n_iteration}, n_samples:{n_training_samples}')
plt.legend()
plt.show()

trainingset = load('trainingset.pkl')
while True:
    index_or_sentene = input('\n\nInsert a sentence:')
    print('\n\n')
    if index_or_sentene == 'quit': break

    try:
        if index_or_sentene.isdigit():
            n_sample = int(index_or_sentene)
            sentence = trainingset[n_sample][0]
            target= trainingset[n_sample][1]
            print(f'sentence: {sentence}')
            print(f'correct: {target}')

            for model in models:
                prediction = model.predict(sentence)
                print(f'{model.name} prediction: {prediction}')
        else:
            print(f'sentence: {index_or_sentene}')
            for model in models:
                prediction = model.predict(index_or_sentene)
                print(f'{model.name} prediction: {prediction}')
    except KeyError:
        print('Please use known words:', *list(dictionary.word2count.keys()) )
        continue