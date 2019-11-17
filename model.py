from utils import *
from layers import *
import numpy as np

"""
transformer-point-network
"""


class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        for k, v in config.items():
            self.__setattr__(k, v)

        self.char_embedding = nn.Embedding(self.vocab_size,self.embedding_size,padding_idx=self.PAD_IDX)
        self.part_embedding = nn.Embedding(self.part_num,self.embedding_size,padding_idx=self.PAD_IDX)

        #max_len + 1 pad
        self.pos_embedding = nn.Embedding(self.max_len + 1,self.embedding_size,padding_idx=self.PAD_IDX)
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_model, self.n_head,self.d_ff, self.dropout) for _ in range(self.layers)])

        self._init_weight()
    def _init_weight(self,scope=1.):
        self.char_embedding.weight.data.uniform_(-scope,scope)
        self.pos_embedding.weight.data.uniform_(-scope,scope)
        self.pos_embedding.weight.data = position(self.max_len + 1, self.d_model)
        self.pos_embedding.weight.requires_grad = False

    def forward(self,encoder_inputs,encoder_parts_inputs,encoder_pos_inputs):
        encoder_input_embedding = self.char_embedding(encoder_inputs)
        encoder_part_embedding = self.part_embedding(encoder_parts_inputs)
        encoder_pos_embedding = self.pos_embedding(encoder_pos_inputs)

        encoder_input = encoder_input_embedding + encoder_part_embedding + encoder_pos_embedding
        self_attn_mask = get_attn_padding_mask(encoder_inputs,encoder_inputs)

        for layer in self.encoder_layers:
            encoder_input = layer(encoder_input,self_attn_mask)

        return encoder_input

class Decoder(nn.Module):
    def __init__(self,config):
        for k, v in config.items():
            self.__setattr__(k, v)
        super(Decoder, self).__init__()   
        self.decoder_embedding = nn.Embedding(self.max_len + 1,self.embedding_size,padding_idx=self.PAD_IDX)
        self.pos_embedding = nn.Embedding(self.max_len + 1,self.embedding_size,padding_idx=self.PAD_IDX)
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.d_model,self.n_head,self.d_ff,self.dropout) for _ in range(self.layers)])

    def _init_weight(self,scope=1.):
        self.decoder_embedding.weight.data.uniform_(-scope,scope)
        self.pos_embedding.weight.data.uniform_(-scope,scope)
        self.pos_embedding.weight.data = position(self.max_len + 1, self.embedding_size)
        self.pos_embedding.weight.requires_grad = False


    def forward(self,encoder_outputs,encoder_inputs,decoder_inputs,decoder_pos):
        decoder_outputs = self.decoder_embedding(decoder_inputs) + self.pos_embedding(decoder_pos)
        #自回归Mask + decoder pad部分mask
        decoder_self_mask = torch.gt(
            get_attn_padding_mask(decoder_inputs,decoder_inputs) + get_attn_subsequent_mask(decoder_inputs),0
        )
        #decoder pad部分, encoder pad部分mask
        decoder_encoder_attn_pad_mask = get_attn_padding_mask(decoder_inputs,encoder_inputs)
        for layer in self.decoder_layers:

            decoder_outputs = layer(decoder_outputs,encoder_outputs,decoder_self_mask,decoder_encoder_attn_pad_mask)

        return decoder_outputs

class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        #测试跑通模型
        self.linear = nn.Linear(config['d_model'],2)
        self.linear_q = nn.Linear(config['d_model'],config['d_model'])
        self.linear_k = nn.Linear(config['d_model'],config['d_model'])
    def attention(self,encoder_outputs,decoder_outputs):
        #利用attention计算分布
        #encoder_outputs, batch*encoder_len*dim
        #decoder_outputs, batch*decoder_len*dim
        attn_q = self.linear_q(decoder_outputs)
        attn_k = self.linear_k(encoder_outputs)
        #batch,decoder_len,encoder_len
        attn_dist =  F.softmax(F.tanh(torch.bmm(attn_q,attn_k.transpose(1,2))),dim=-1)

        return attn_dist

    def forward(self,encoder_inputs,encoder_parts,decoder_inputs,encoder_pos,decoder_pos):
        encoder_len = encoder_inputs.size(1)

        encoder_outputs = self.encoder(encoder_inputs,encoder_parts,encoder_pos)
        decoder_outputs = self.decoder(encoder_outputs,encoder_inputs,decoder_inputs,decoder_pos)

        #point network 利用attention计算概率分布
        attention_dist = self.attention(encoder_outputs,decoder_outputs)
        return attention_dist.view(-1,encoder_len)


if __name__ == '__main__':

    config = {
        'embedding_size':10,
        'd_model':10,
        'n_head':2,
        'd_ff':10,
        'vocab_size':10,
        'max_len':4,
        'part_num':4,
        'PAD_IDX':0,
        'layers':1,
        'dropout':0.1
    }

    encoder_inputs = torch.tensor(np.array([[1,2,3,4,0,0],
                                           [2,3,4,5,0,0]]))
    encoder_parts = torch.tensor(np.array([[1,1,2,3,0,0],
                                           [1,1,2,3,0,0]]))
    encoder_pos = torch.tensor(np.array([[1,2,3,4,0,0],
                                           [1,2,3,4,0,0]]))

    decoder_inputs = torch.tensor(np.array([[1, 2, 3, 4, 0, 0],
                                           [2, 3, 4, 5, 0, 0]]))
    decoder_outputs = torch.tensor(np.array([[1, 1, 2, 3, 0, 0],
                                          [1, 1, 2, 3, 0, 0]]))
    decoder_pos = torch.tensor(np.array([[1, 2, 3, 4, 0, 0],
                                        [1, 2, 3, 4, 0, 0]]))

    model = Transformer(config)
    outputs = model(encoder_inputs,encoder_parts,decoder_inputs,encoder_pos,decoder_pos)

    print(outputs.size())