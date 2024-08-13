


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 15
derWords1 = {0: 'SOS', 1: 'EOS', 2: '-', 3 : '7', 4 : '*', 5: 'cos', 6 : '(', 7: 'x', 8: ')', 9: '0',
                 10: '1', 11: '2', 12: '6', 13: '/', 14: '+', 15: 'log', 16: '3', 17: '5', 18: 'sin'
                 , 19: '9', 20: '4', 21: '8', 22: 'tan'}

functionWords1 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '7', 4: '*', 5: 'sin', 6: '(', 7: 'x', 8: ')', 9: '4', 10: '2',
            11: '+', 12: '6', 13: 'atan', 14: 'log', 15: '/', 16: '3', 17: '1', 18: '0', 19: '5',
            20: '9', 21: 'cos', 22: '8', 23: 'tan'}

dWords2 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '4', 4: '/', 5: '(', 6: 'x', 7: '*', 8: '2', 9: '+',
           10: '1', 11: ')', 12: '6', 13: 'sqrt', 14: '0', 15: '5', 16: '3', 17: 'log', 18: '7',
          19: '8', 20: 'cos', 21: 'tan', 22: '9', 23: 'exp', 24: 'sec', 25: 'sin'}

fWords2 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '4', 4: '*', 5: 'atan', 6: '(', 7: 'x', 8: ')',
                9: '6', 10: 'asin', 11: '3', 12: '0', 13: '2', 14: '8', 15: '/', 16: '9', 17: '7',
           18: 'log', 19: '5', 20: '1', 21: 'sin', 22: '+', 23: 'sec', 24: 'tan', 25: 'exp'}



fWords3 = {0: 'SOS', 1: 'EOS', 2: '6', 3: '*', 4: 'x', 5: '3', 6: '+', 7: '7', 8: '-', 9: '4', 10: 'log', 11: '(',
            12: ')', 13: '/', 14: '2', 15: '1', 16: '5', 17: 'atan', 18: '9', 19: '8', 20: '0', 21: 'sin', 22: 'tan',
           23: 'asin', 24: 'sec', 25: 'cos', 26: 'zoo', 27: 'exp', 28: 'nan'}
dWords3 = {0: 'SOS', 1: 'EOS', 2: '1', 3: '8', 4: '*', 5: 'x', 6: '2', 7: '+', 8: '-', 9: '4', 10: '/',
            11: '(', 12: 'log', 13: ')', 14: '0', 15: '5', 16: '9', 17: '7', 18: '3', 19: '6', 20: 'cos', 21: 'tan',
           22: 'sqrt', 23: 'sec', 24: 'sin', 25: 'exp', 26: 'zoo'}



fWords4 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '7', 4: '*', 5: 'tan', 6: '(', 7: '3', 8: 'x', 9: ')',
10: '1', 11: '2', 12: '9', 13: '5', 14: 'sec', 15: '6', 16: '0', 17: '8', 18: 'exp', 19: 'log',
20: '/', 21: '4', 22: 'cos', 23: '+', 24: 'sin', 25: 'atan', 26: 'asin'}

dWords4 ={0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '1', 5: '*', 6: 'tan', 7: '(', 8: '3', 9: 'x',
10: ')', 11: '6', 12: '0', 13: '5', 14: 'sec', 15: '4', 16: 'exp', 17: 'log', 18: '7', 19: '8',
20: '/', 21: '+', 22: 'sin', 23: 'cos', 24: 'sqrt', 25: '9', 26: 'atan', 27: 'asin'}


fWords5 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '/', 5: 'x', 6: '*', 7: '4', 8: '1', 9: '3',
10: '8', 11: '9', 12: '+', 13: '6', 14: '0', 15: '7', 16: '(', 17: ')', 18: '5', 19: 'sin',
20: 'cos', 21: 'log', 22: 'tan', 23: 'atan', 24: 'asin', 25: 'sec'}

dWords5 = {0: 'SOS', 1: 'EOS', 2: '4', 3: '/', 4: 'x', 5: '*', 6: '3', 7: '-',
8: '2', 9: '+', 10: '1', 11: '8', 12: '6', 13: '7', 14: '0', 15: '5', 16: '9',
17: '(', 18: ')', 19: 'sin', 20: 'cos', 21: 'log', 22: 'tan', 23: 'sqrt', 24: 'sec'}




fWords6 = {0: 'SOS', 1: 'EOS', 2: '4', 3: '/', 4: 'x', 5: '*', 6: '3', 7: '-', 8: '2', 9: '+', 10: '1', 11: '8', 12: '6', 13: '7', 14: '0', 15: '5', 16: '9', 17: '(', 18: ')', 19: 'sin', 20: 'cos', 21: 'log', 22: 'tan', 23: 'sqrt', 24: 'sec'}

dWords6 ={0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '/', 5: 'x', 6: '*', 7: '4', 8: '1', 9: '3', 10: '8', 11: '9', 12: '+', 13: '6', 14: '0', 15: '(', 16: ')', 17: '5', 18: '7', 19: 'sin', 20: 'cos', 21: 'log', 22: 'tan', 23: 'atan', 24: 'asin', 25: 'sec'}


dWords7 = {0: 'SOS', 1: 'EOS', 2: '2', 3: '0', 4: '-', 5: '3', 6: '*', 7: 'x', 8: '4',
9: '8', 10: '7', 11: '1', 12: '/', 13: '(', 14: 'log', 15: '6', 16: ')', 17: '5',
18: 'tan', 19: '+', 20: 'sin', 21: 'cos', 22: 'sqrt', 23: '9', 24: 'sec'}

fWords7 = {0: 'SOS', 1: 'EOS', 2: '2', 3: '*', 4: 'x', 5: '+', 6: '1', 7: '3',
8: '-', 9: '6', 10: '8', 11: '0', 12: 'log', 13: '(', 14: ')', 15: '/', 16: '5',
17: '7', 18: '4', 19: 'tan', 20: 'cos', 21: 'sin', 22: 'atan', 23: 'asin', 24: '9', 25: 'sec'}



fWords8 = {0: 'SOS', 1: 'EOS', 2: '2', 3: '*', 4: 'x', 5: '+', 6: '1', 7: '3', 8: '-', 9: '6', 10: '8', 11: '0', 12: 'log', 13: '(', 14: ')', 15: '/', 16: '5', 17: '7', 18: '4', 19: 'tan', 20: 'cos', 21: 'sin', 22: 'atan', 23: 'asin', 24: '9', 25: 'sec'}
dWords8 ={0: 'SOS', 1: 'EOS', 2: '2', 3: '0', 4: '-', 5: '3', 6: '*', 7: 'x', 8: '4', 9: '8', 10: '7', 11: '1', 12: '/', 13: '(', 14: 'log', 15: '6', 16: ')', 17: '5', 18: 'tan', 19: '+', 20: 'sin', 21: 'cos', 22: 'sqrt', 23: '9', 24: 'sec'}



fWords9 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '3', 4: '2', 5: 'sec', 6: '(', 7: 'x', 8: ')', 9: '5', 10: '*', 11: '^', 12: '6', 13: '+', 14: '/', 15: '7', 16: '1', 17: '0', 18: '4', 19: '8', 20: 'cos', 21: 'asin', 22: 'tan', 23: '9', 24: 'log', 25: 'atan', 26: 'sin', 27: 'exp'}
dWords9 = {0: 'SOS', 1: 'EOS', 2: '0', 3: '-', 4: 'tan', 5: '(', 6: 'x', 7: ')', 8: '*', 9: 'sec', 10: '3', 11: '^', 12: '5', 13: '/', 14: '7', 15: '2', 16: '1', 17: '4', 18: 'sin', 19: 'sqrt', 20: '8', 21: '6', 22: 'log', 23: '9', 24: '+', 25: 'cos', 26: 'exp', 27: 'atan'}


fWords10 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '5', 5: '*', 6: 'log(', 7: 'x', 8: ')', 9: '/', 10: '6', 11: '7', 12: 'asin(', 13: '^', 14: '9', 15: '3', 16: '+', 17: '1', 18: '4', 19: 'cos(', 20: 'sin(', 21: 'atan(', 22: '8', 23: '(', 24: '0', 25: 'exp(', 26: 'sec(', 27: 'tan('}
dWords10 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '5', 5: '/', 6: '(', 7: 'x', 8: '*', 9: 'log(', 10: '6', 11: ')', 12: '7', 13: 'sqrt(', 14: '1', 15: '^', 16: '4', 17: '0', 18: 'sin(', 19: 'cos(', 20: '8', 21: '+', 22: '3', 23: 'exp(', 24: 'tan(', 25: 'sec(', 26: '9', 27: 'atan(', 28: 'asin('}

fWords11 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '5', 5: '*', 6: 'log(', 7: 'x', 8: ')', 9: '/', 10: '6', 11: '7', 12: 'asin(', 13: '^', 14: '9', 15: '3', 16: '+', 17: '1', 18: '4', 19: 'cos(', 20: 'sin(', 21: 'atan(', 22: '8', 23: '(', 24: '0', 25: 'exp(', 26: 'sec(', 27: 'tan('}
dWords11 ={0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '5', 5: '/', 6: '(', 7: 'x', 8: '*', 9: 'log(', 10: '6', 11: ')', 12: '7', 13: 'sqrt(', 14: '1', 15: '^', 16: '4', 17: '0', 18: 'sin(', 19: 'cos(', 20: '8', 21: '+', 22: '3', 23: 'exp(', 24: 'tan(', 25: 'sec(', 26: '9', 27: 'atan(', 28: 'asin('}
fWords12 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '5', 5: '*', 6: 'log(', 7: 'x', 8: ')', 9: '/', 10: '6', 11: '7', 12: 'asin(', 13: '^', 14: '9', 15: '3', 16: '+', 17: '1', 18: '4', 19: 'cos(', 20: 'sin(', 21: 'atan(', 22: '8', 23: '(', 24: '0', 25: 'exp(', 26: 'sec(', 27: 'tan('}
dWords12 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '2', 4: '5', 5: '/', 6: '(', 7: 'x', 8: '*', 9: 'log(', 10: '6', 11: ')', 12: '7', 13: 'sqrt(', 14: '1', 15: '^', 16: '4', 17: '0', 18: 'sin(', 19: 'cos(', 20: '8', 21: '+', 22: '3', 23: 'exp(', 24: 'tan(', 25: 'sec(', 26: '9', 27: 'atan(', 28: 'asin('}
fWords13 = {0: 'SOS', 1: 'EOS', 2: '7', 3: '2', 4: '3', 5: '-', 6: '*', 7: 'sin(', 8: 'x', 9: ')', 10: '1', 11: 'log(', 12: '/', 13: '6', 14: '0', 15: '8', 16: '^', 17: '+', 18: '4', 19: '9', 20: '5', 21: 'exp(', 22: 'sec(', 23: 'atan(', 24: 'cos(', 25: 'tan(', 26: '(', 27: 'asin('}
dWords13 ={0: 'SOS', 1: 'EOS', 2: '0', 3: '-', 4: '3', 5: '2', 6: '*', 7: 'cos(', 8: 'x', 9: ')', 10: '/', 11: '(', 12: 'log(', 13: '6', 14: '1', 15: '^', 16: '+', 17: '5', 18: '4', 19: 'exp(', 20: '7', 21: 'tan(', 22: 'sec(', 23: '8', 24: 'sin(', 25: '9', 26: 'sqrt(', 27: 'asin(', 28: 'atan('}
fWords14 = {0: 'SOS', 1: 'EOS', 2: '8', 3: '5', 4: '-', 5: '2', 6: '*', 7: 'x', 8: '^', 9: '+', 10: '1', 11: '0', 12: 'cos(', 13: ')', 14: '7', 15: '6', 16: '3', 17: '/', 18: '4', 19: 'exp(', 20: 'log(', 21: 'sin(', 22: '9', 23: '(', 24: 'tan(', 25: 'sec(', 26: 'asin(', 27: 'atan('}
dWords14 = {0: 'SOS', 1: 'EOS', 2: '0', 3: '2', 4: '-', 5: '4', 6: '*', 7: 'x', 8: '1', 9: 'sin(', 10: ')', 11: '9', 12: '^', 13: '6', 14: '+', 15: '3', 16: '5', 17: '7', 18: 'exp(', 19: 'log(', 20: '/', 21: '(', 22: 'cos(', 23: '8', 24: 'tan(', 25: 'sec(', 26: 'sqrt(', 27: 'atan(', 28: 'asin('}
fWords15 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '6', 4: '*', 5: 'log(', 6: 'x', 7: ')', 8: '/', 9: '5', 10: '^', 11: '4', 12: '2', 13: '+', 14: 'sin(', 15: '8', 16: '1', 17: '3', 18: 'cos(', 19: '9', 20: 'sec(', 21: '7', 22: '0', 23: 'exp(', 24: 'tan(', 25: 'asin(', 26: 'atan(', 27: '('}
dWords15 = {0: 'SOS', 1: 'EOS', 2: '-', 3: '6', 4: '/', 5: '(', 6: 'x', 7: '*', 8: 'log(', 9: ')', 10: '2', 11: '0', 12: '^', 13: '3', 14: '1', 15: '+', 16: '5', 17: 'cos(', 18: 'sin(', 19: '8', 20: '7', 21: '9', 22: 'tan(', 23: 'sec(', 24: '4', 25: 'exp(', 26: 'sqrt(', 27: 'atan(', 28: 'asin('}



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
    output = ''.join(outputs)
    print(output)


def evalSimilarity(encoder, s, e):
    s1 = tensorFromSentence(s)
    s2 = tensorFromSentence(e)
    outputs1, h1 = encoder(s1)
    outputs2, h2 = encoder(s2)
    similarity = cosine_similarity(h1[-1], h2[-1])
    sim = similarity.item()
    distance = torch.dist(h1[-1], h2[-1])
    print(distance.item())
    print(sim)
    return sim, distance.item()

derWords = dWords13
functionWords = fWords13
swappedDer = {value: key for key, value in derWords.items()}
swappedFunc = {value: key for key, value in functionWords.items()}

if __name__ == "__main__":
  
    hidden_size = 180
    batch_size = 32

    PATH = './encoder13_netpth.'
    PATH2 = './decoder13_netpth.'
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

    while(True):
        func = input("Function: ")
        func2 = input("Function: ")
        evalSimilarity(encoder, func, func2)
        stopper = input("Stop? ")
        if stopper == 'y':
            break
