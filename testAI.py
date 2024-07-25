from runAI2 import  *
from DerivativeAI import readLangs, get_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F


def classifyFailed(f, data):
    if "trig" in f:
        data[0] += 1
    if "poly" in f:
        data[1] += 1
    if "exp" in f:
        data[2] += 1
       
    if "log" in f:
        data[3] += 1

    if "arc" in f:
        data[4] += 1
    if "const" in f:
        data[5] += 1
    if "neg" in f:
        data[6] += 1
    if "unknown" in f:
        data[7] += 1



hidden_size = 128
batch_size = 32   
_, _, pairs, dataLoader = get_dataloader(batch_size, False, "testingF.txt", "testingD.txt")

correct = 0
PATH = './encoder5Ver2_netpth.'
PATH2 = './decoder5Ver2_netpth.'
encoder = EncoderRNN(len(functionWords), hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, len(derWords)).to(device)
typesFile = open('testingTypes.txt', encoding='utf-8').readlines()
encoder.load_state_dict(torch.load(PATH))
decoder.load_state_dict(torch.load(PATH2))
failedArray = [0] * 8
encoder.eval()
decoder.eval()
"""

a = 6
outputs, _ = evaluate (encoder, decoder, pairs[a][0] )
output = ' '.join(outputs)
print(pairs[a][0])
print(pairs[a][1])
print(output)
with open('failedFunc.txt', 'w') as f1:
    pass

with open('failedFunc.txt', 'a') as f1:
    for i, pair in enumerate(pairs):
        d = pair[1]
    
        try:
            outputs, _ = evaluate(encoder, decoder, pair[0])
            dString = ""
            for word in d:
                dString +=word
            dString += " <EOS>"
            output = ' '.join(outputs)
            if output == dString:
                correct += 1
            else:
                f1.write(i + "\n")
                classifyFailed(typesFile[i], failedArray)
                
        except Exception as e:
            f1.write(str(i) + "\n")
            classifyFailed(typesFile[i], failedArray)

print("trig, poly, exp, log, arc, const, neg, unknown\n")
print(failedArray)
print("Percentage correct = " + str(100*correct/len(typesFile)))

 """


