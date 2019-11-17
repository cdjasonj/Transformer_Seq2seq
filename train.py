from model import *
from data_loader import data_generator
import torch
from tqdm import tqdm

config = {
    'data_path':'./corpus.txt',
    'batch_size':64,
    'epoch':1,
    'embedding_size': 100,
    'd_model': 512,
    'n_head': 8,
    'd_ff': 1024,
    'part_num': 4,
    'PAD_IDX': 0, #encoder decoder Inputs pad
    'layers': 6,
    'dropout': 0.1
}

DataGenerator = data_generator(config['data_path'],config['batch_size'])
config['vocab_size'] = len(DataGenerator.char2id)
config['max_len'] = DataGenerator.max_len + 4


#############
#build model
###########
model = Transformer(config)
model = model.cuda()
optimizer = torch.optim.Adam(model.get_trainable_parameters())
crit = torch.nn.CrossEntropyLoss(size_average=False)

###########
#train model
###########

def get_losses(gold,pred):
    #返回loss
    gold = gold.contiguous().view(-1)
    loss = crit(pred, gold)

    return loss


def train():
    model.train()
    for encoder_inputs, encoder_part_inputs, decoder_inputs, \
        decoder_outputs, encoder_pos, decoder_pos in DataGenerator.train_batcher():
        optimizer.zero_grad()
        # encoder_inputs, encoder_parts, decoder_inputs, encoder_pos, decoder_pos
        pred = model(encoder_inputs, encoder_part_inputs, decoder_inputs, encoder_pos, decoder_pos)
        loss = get_losses(decoder_outputs, pred)
        loss.backward()
        optimizer.step()

        print(loss)

def eval():
    pass

def pred():
    pass
