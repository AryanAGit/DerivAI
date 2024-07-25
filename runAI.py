


import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 15
"""
dWords2 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '4', 4: '/', 5: '(', 6: 'x', 7: '*', 8: '2', 9: '+',
           10: '1', 11: ')', 12: '6', 13: 'sqrt', 14: '0', 15: '5', 16: '3', 17: 'log', 18: '7',
          19: '8', 20: 'cos', 21: 'tan', 22: '9', 23: 'exp', 24: 'sec', 25: 'sin'}

fWords2 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '4', 4: '*', 5: 'atan', 6: '(', 7: 'x', 8: ')',
                9: '6', 10: 'asin', 11: '3', 12: '0', 13: '2', 14: '8', 15: '/', 16: '9', 17: '7',
           18: 'log', 19: '5', 20: '1', 21: 'sin', 22: '+', 23: 'sec', 24: 'tan', 25: 'exp'}
"""

"""
fWords3 = {0: 'SOS', 1: 'EOS', 2: '6', 3: '*', 4: 'x', 5: '3', 6: '+', 7: '7', 8: '-', 9: '4', 10: 'log', 11: '(',
            12: ')', 13: '/', 14: '2', 15: '1', 16: '5', 17: 'atan', 18: '9', 19: '8', 20: '0', 21: 'sin', 22: 'tan',
           23: 'asin', 24: 'sec', 25: 'cos', 26: 'zoo', 27: 'exp', 28: 'nan'}
derWords3 = {0: 'SOS', 1: 'EOS', 2: '1', 3: '8', 4: '*', 5: 'x', 6: '2', 7: '+', 8: '-', 9: '4', 10: '/',
            11: '(', 12: 'log', 13: ')', 14: '0', 15: '5', 16: '9', 17: '7', 18: '3', 19: '6', 20: 'cos', 21: 'tan',
           22: 'sqrt', 23: 'sec', 24: 'sin', 25: 'exp', 26: 'zoo'}
"""

"""
fWords4 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '7', 4: '*', 5: 'tan', 6: '(', 7: '3', 8: 'x', 9: ')',
10: '1', 11: '2', 12: '9', 13: '5', 14: 'sec', 15: '6', 16: '0', 17: '8', 18: 'exp', 19: 'log',
20: '/', 21: '4', 22: 'cos', 23: '+', 24: 'sin', 25: 'atan', 26: 'asin'}

derWords4 ={0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '1', 5: '*', 6: 'tan', 7: '(', 8: '3', 9: 'x',
10: ')', 11: '6', 12: '0', 13: '5', 14: 'sec', 15: '4', 16: 'exp', 17: 'log', 18: '7', 19: '8',
20: '/', 21: '+', 22: 'sin', 23: 'cos', 24: 'sqrt', 25: '9', 26: 'atan', 27: 'asin'}
"""
"""
fWords5 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '/', 5: 'x', 6: '*', 7: '4', 8: '1', 9: '3',
10: '8', 11: '9', 12: '+', 13: '6', 14: '0', 15: '7', 16: '(', 17: ')', 18: '5', 19: 'sin',
20: 'cos', 21: 'log', 22: 'tan', 23: 'atan', 24: 'asin', 25: 'sec'}

dword5 = {0: 'SOS', 1: 'EOS', 2: '4', 3: '/', 4: 'x', 5: '*', 6: '3', 7: '-',
8: '2', 9: '+', 10: '1', 11: '8', 12: '6', 13: '7', 14: '0', 15: '5', 16: '9',
17: '(', 18: ')', 19: 'sin', 20: 'cos', 21: 'log', 22: 'tan', 23: 'sqrt', 24: 'sec'}
"""


"""  
fwords6 = {0: 'SOS', 1: 'EOS', 2: '4', 3: '/', 4: 'x', 5: '*', 6: '3', 7: '-', 8: '2', 9: '+', 10: '1', 11: '8', 12: '6', 13: '7', 14: '0', 15: '5', 16: '9', 17: '(', 18: ')', 19: 'sin', 20: 'cos', 21: 'log', 22: 'tan', 23: 'sqrt', 24: 'sec'}

dwords6 ={0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '/', 5: 'x', 6: '*', 7: '4', 8: '1', 9: '3', 10: '8', 11: '9', 12: '+', 13: '6', 14: '0', 15: '(', 16: ')', 17: '5', 18: '7', 19: 'sin', 20: 'cos', 21: 'log', 22: 'tan', 23: 'atan', 24: 'asin', 25: 'sec'}
"""
"""
dwords7 = {0: 'SOS', 1: 'EOS', 2: '2', 3: '0', 4: '-', 5: '3', 6: '*', 7: 'x', 8: '4',
9: '8', 10: '7', 11: '1', 12: '/', 13: '(', 14: 'log', 15: '6', 16: ')', 17: '5',
18: 'tan', 19: '+', 20: 'sin', 21: 'cos', 22: 'sqrt', 23: '9', 24: 'sec'}

Fwords7 = {0: 'SOS', 1: 'EOS', 2: '2', 3: '*', 4: 'x', 5: '+', 6: '1', 7: '3',
8: '-', 9: '6', 10: '8', 11: '0', 12: 'log', 13: '(', 14: ')', 15: '/', 16: '5',
17: '7', 18: '4', 19: 'tan', 20: 'cos', 21: 'sin', 22: 'atan', 23: 'asin', 24: '9', 25: 'sec'}
"""


fWords8 = {0: 'SOS', 1: 'EOS', 2: '2', 3: '*', 4: 'x', 5: '+', 6: '1', 7: '3', 8: '-', 9: '6', 10: '8', 11: '0', 12: 'log', 13: '(', 14: ')', 15: '/', 16: '5', 17: '7', 18: '4', 19: 'tan', 20: 'cos', 21: 'sin', 22: 'atan', 23: 'asin', 24: '9', 25: 'sec'}
dWords8 ={0: 'SOS', 1: 'EOS', 2: '2', 3: '0', 4: '-', 5: '3', 6: '*', 7: 'x', 8: '4', 9: '8', 10: '7', 11: '1', 12: '/', 13: '(', 14: 'log', 15: '6', 16: ')', 17: '5', 18: 'tan', 19: '+', 20: 'sin', 21: 'cos', 22: 'sqrt', 23: '9', 24: 'sec'}

derWords = dWords8
functionWords = fWords8
swappedDer = {value: key for key, value in derWords.items()}
swappedFunc = {value: key for key, value in functionWords.items()}
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
def indexesFromSentence( sentence):
    return [swappedFunc[word] for word in sentence.split(' ')]

def tensorFromSentence( sentence):
    indexes = indexesFromSentence( sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)



def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    showAttention(input_sentence, output_words, attentions[0, :len(output_words), :])



def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence( sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(derWords[idx.item()])
    return decoded_words, decoder_attn


def evalUser(encoder, decoder, s):
    outputs, _= evaluate(encoder, decoder, s)
    output = ' '.join(outputs)
    print(output)

if __name__ == "__main__":
  
    hidden_size = 200
    batch_size = 32

    PATH = './encoder8_netpth.'
    PATH2 = './decoder8_netpth.'
    encoder = EncoderRNN(len(functionWords), hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, len(derWords)).to(device)

    encoder.load_state_dict(torch.load(PATH))
    decoder.load_state_dict(torch.load(PATH2))

    encoder.eval()
    decoder.eval()
    torch.no_grad()
    while(True):
        func = input("Function: ")
        evalUser(encoder, decoder, func)
        evaluateAndShowAttention(func)
        stopper = input("Stop? ")
        if stopper == 'y':
            break

