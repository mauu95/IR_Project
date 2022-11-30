import json
from utility import *
from parameters import *

def readjsonl(nlines, path):
    with open(path, 'r') as f:
        json_list = list(f)
        f.close()

    i = 0
    for json_str in json_list:
        if i>=nlines: break 
        jline = json.loads(json_str)
        if jline['gold_label'] != 'entailment':continue
        if len(word_tokenize(jline['sentence1'])) >= sentence_maxlen:continue
        i += 1
        
        yield (jline['sentence1'],jline['sentence2'])

path = '.\\..\\Dataset\\snli_1.0\\'
trainingset = list(readjsonl(n_training_samples, path + 'snli_1.0_train.jsonl'))
testset = list(readjsonl(10, path + 'snli_1.0_test.jsonl'))
devset = list(readjsonl(10, path + 'snli_1.0_dev.jsonl'))

store(trainingset, 'trainingset')

