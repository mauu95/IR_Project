import torch
import torch.nn as nn
from utility import *
from dictionary import *
from torch import optim
import random
import torch.optim.lr_scheduler as lr_scheduler
from components import *

name = 'Transformer'

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = Residual(MultiHeadAttention(6, hidden_size, hidden_size, hidden_size), hidden_size)
        self.feedforward = Residual( feed_forward(hidden_size), hidden_size )

    def init_hidden_layer(self):
        return torch.zeros(1,1,hidden_size, device=device)

    def forward(self, input):
        val = self.embedding(input)
        val.add_(position_encoding(val.shape[0], hidden_size, device=device))
        val = val.view(1,-1,hidden_size)
        val = self.attention(val, val, val)
        val = self.feedforward(val)
        return val


class Decoder(nn.Module):
    def __init__(self, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = Residual(MultiHeadAttention(6, hidden_size, hidden_size, hidden_size), hidden_size)
        self.attention2 = Residual(MultiHeadAttention(6, hidden_size, hidden_size, hidden_size), hidden_size)
        self.feedforward = Residual(feed_forward(hidden_size), hidden_size)

    def forward(self, tgt, memory):
        val = self.embedding(tgt)
        val = val.view(-1,1,hidden_size)
        val.add_(position_encoding(val.shape[0], hidden_size, device=device))
        val = val.view(1,-1,hidden_size)
        val = self.attention(val, val, val)
        val = self.attention2(val[0][-1].view(1,1,-1), memory, memory)
        val = self.feedforward(val)
        val = self.linear(val[0])
        val = self.softmax(val)
        return val

# PARAMETERS
trainingset = load('trainingset.pkl')
dictionary = load('dictionary.pkl')
encoder = Encoder(dictionary.n_words).to(device)
decoder = Decoder(dictionary.n_words).to(device)
enc_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
dec_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

enc_scheduler = lr_scheduler.ReduceLROnPlateau(enc_optimizer, patience=patience, factor=scheduler_factor)
dec_scheduler = lr_scheduler.ReduceLROnPlateau(dec_optimizer, patience=patience, factor=scheduler_factor)

# TRAIN
def train(sentence, target):
    sentence = Sentence2Tensor(dictionary, sentence)
    target = Sentence2Tensor(dictionary, target)

    enc_hid = encoder(sentence)

    dec_in = torch.tensor([SOS_token], device=device)
    dec_hid = enc_hid
    loss = 0

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for word in target:
            dec_out = decoder(dec_in, dec_hid)
            loss += criterion(dec_out, word)
            dec_in = torch.cat([dec_in, word])
    else:
        for word in target:
            dec_out = decoder(dec_in, dec_hid)
            loss += criterion(dec_out, word)
            predicted_word = Indexes2Tensor(dec_out.topk(1)[1].squeeze().detach().item())[0]
            dec_in = torch.cat([dec_in, predicted_word])

    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    if use_scheduler:
        enc_scheduler.step(loss)
        dec_scheduler.step(loss)
    return loss.item() / len(target)

def train_iters():
    training_pairs = [random.choice(trainingset) for i in range(n_iteration)]

    hist = []
    i = 0
    for training_pair in training_pairs:
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor)
        if i % (n_iteration//training_status_n_msg) == 0:
            print(f'Training {100/n_iteration * i:.1f}% complete. Loss = {loss}')
        hist.append(loss)
        i+=1
    return hist

# TEST
def predict(sentence):
    with torch.no_grad():
        sentence = Sentence2Tensor(dictionary, sentence)
        
        enc_hid = encoder(sentence)

        dec_in = torch.tensor([SOS_token], device=device)
        dec_hid = enc_hid

        predicted_words = []
        for i in range(sentence_maxlen):
            dec_out = decoder(dec_in, dec_hid)
            predicted_word = Indexes2Tensor(dec_out.topk(1)[1].squeeze().detach().item())[0]
            dec_in = torch.cat([dec_in, predicted_word])
            if predicted_word.item() == EOS_token: 
                break
            predicted_words.append(dictionary.index2word[predicted_word.item()])
        return ' '.join(predicted_words)


if __name__ == '__main__':
    train_iters()
    while True:
        index_or_sentene = input('\n\nInsert a sentence:')
        print('\n\n')
        if index_or_sentene == 'quit': break

        try:
            if index_or_sentene.isdigit():
                n_sample = int(index_or_sentene)
                sentence = trainingset[n_sample][0]
                target= trainingset[n_sample][1]
                prediction = predict(sentence)
                print(f'sentence: {sentence}')
                print(f'correct: {target}')
                print(f'prediction: {prediction}')
            else:
                prediction = predict(index_or_sentene)
                print(f'sentence: {index_or_sentene}')
                print(f'prediction: {prediction}')
        except KeyError:
            print('Please use known words:', *list(dictionary.word2count.keys()) )
            continue