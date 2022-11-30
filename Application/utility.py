import pickle
from parameters import *
import os
import nltk

def store(obj, filename):
    directory = f'.\\{save_folder}'
    file = filename + '.pkl'
    path = f'{directory}\\{file}'

    if not os.path.exists(directory): os.mkdir(directory)
    
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()

def load(filename):
    directory = f'.\\{save_folder}'
    path = f'{directory}\\{filename}'

    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def word_tokenize(sentence):
    return nltk.tokenize.word_tokenize(sentence)