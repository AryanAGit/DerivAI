
from modelFuncs import *
from langFuncs import *
from model import *


if __name__ == "__main__":

   
    numEpochs = 100

    flip = False
    hidden_size = 128
    batch_size = 32
    
    input_lang, output_lang, pairs, train_dataloader = get_dataloader(batch_size, flip, 'testingF.txt', 'testingD.txt')
    _, _, _, equalLoader = get_dataloader(batch_size, flip, 'testingF.txt', 'equals.txt')
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train2(train_dataloader, equalLoader, encoder, decoder, numEpochs, print_every=5, plot_every=5)


    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang)

    
    PATH = './encoder15_netpth.'
    PATH2 = './decoder15_netpth.'
    torch.save(encoder.state_dict(), PATH)
    torch.save(decoder.state_dict(), PATH2)
    print("Input: ")
    print(input_lang.index2word)
    print('\nOutput: ')
    print(output_lang.index2word)
    
    evaluateAndShowAttention('x ^ 2',encoder, decoder, input_lang, output_lang)
    evaluateAndShowAttention('x + 3',encoder, decoder, input_lang, output_lang)
    evaluateAndShowAttention('sin( x )',encoder, decoder, input_lang, output_lang)
    for i in range(5):
        evalUser(encoder, decoder, input_lang, output_lang)



