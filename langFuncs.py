import re
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 15

def tokenize(f):
    # r'\d|\w+\(|\w+|[^\s\w]'
    token_pattern = re.compile(r'\d|\w+\(|\w+|[^\s\w]')
    tokens = token_pattern.findall(f)
    function = str(' '.join(tokens))
   # function = function.replace('^', "* *")
    return function

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False, f1 = "functionToken.txt", f2 = "dxToken.txt"):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(f1, encoding='utf-8').readlines()
    lines2 = open(f2, encoding = 'utf-8').readlines()
    pairs = []
    print("length = " + str(len(lines)))
    # Split every line into pairs and normalize
    for i in range(len(lines)):
        pair = []
        f = lines[i].strip()
        d = lines2[i].strip()
        function = tokenize(f)
        dx = tokenize(d)
        pair.append(function)
        pair.append(dx)
        pairs.append(pair)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs




def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and p[0] != "0"


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False, f1 = "functionToken.txt", f2 = "dxToken.txt"):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, f1, f2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)
