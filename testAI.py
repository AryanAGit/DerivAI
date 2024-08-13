from runAI import  *
from modelFuncs import get_dataloader
from langFuncs import readLangs
import torch
import torch.nn as nn
import torch.nn.functional as F


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



hidden_size =128
batch_size = 32   
_, _, pairs, dataLoader = get_dataloader(batch_size, False, "testing3F.txt", "testing3D.txt")

correct = 0
PATH = './encoder15_netpth.'
PATH2 = './decoder15_netpth.'
encoder = EncoderRNN(len(functionWords), hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, len(derWords)).to(device)
typesFile = open('testingTypes.txt', encoding='utf-8').readlines()
encoder.load_state_dict(torch.load(PATH))
decoder.load_state_dict(torch.load(PATH2))
failedArray = [0] * 8
encoder.eval()
decoder.eval()

a = 0
outputs, _ = evaluate (encoder, decoder, pairs[a][0] )
output = ''.join(outputs)
b = pairs[a][1]
c = b.replace(' ', '')
c+="<EOS>"
print(pairs[a][0])
print(c)
print(output)
with open('failedFunc.txt', 'w') as f1:
    pass

with open('failedFunc.txt', 'a') as f1:
    for i, pair in enumerate(pairs):
        d = pair[1]
    
        try:
            outputs, _ = evaluate(encoder, decoder, pair[0])
            dString = d.replace(' ', '')
            dString += "<EOS>"
            output = ''.join(outputs)
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




