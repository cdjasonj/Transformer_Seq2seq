import numpy as np
import random
import torch
from torch.autograd import Variable


# pos embedding .需要max len
# decoder embedding 需要max len

class data_generator():

    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.datas = self.loadData()
        self.train_index, self.dev_index = self.split_datas()
        self.id2char, self.char2id = self.get_dic()
        self.batch_size = batch_size

        self.max_len = max([len(data[0] + data[1] + data[2]) for data in self.datas])

    def loadData(self):
        datas = []
        with open(self.data_path, encoding='utf-8') as fr:
            for line in fr:
                context1, context2, uttrance = line.strip().split('\t')[0], line.strip().split('\t')[2], line.strip().split('\t')[4]
                truth = line.strip().split('\t')[-1]
                datas.append([context1, context2, uttrance, truth])
        return datas

    def get_dic(self):
        char_dict = {}
        for data in self.datas:
            for sentence in data:
                for char in sentence:
                    char_dict[char] = char_dict.get(char, 0) + 1

        char_dict = {i: j for i, j in char_dict.items()}
        id2char = {i + 4: j for i, j in enumerate(char_dict)}  # 0 pad , 1 UNK , 2 START 3, END
        char2id = {j: i for i, j in id2char.items()}

        return id2char, char2id

    def split_datas(self):
        # 划分数据集
        index = list(range(len(self.datas)))
        np.random.shuffle(index)
        train_index = random.sample(index, round(0.8 * len(index)))
        dev_index = set(index) - set(train_index)
        return train_index, dev_index

    def seq_padding(self, texts):

        # 按照batch最长进行pad
        maxlen = max([len(text) for text in texts])
        return [x + [0] * (maxlen - len(x)) if len(x) < maxlen else x[:maxlen] for x in texts]

    def get_decoder_output(self, context1, context2, uttrance, truth):
        decoder_output = [-1] * len(truth)
        encoder_input = context1 + context2 + uttrance
        # 找 truth 在 uttrance 里面的下标
        origin_question = ''
        for char in truth:
            if char in uttrance:
                origin_question += char

        # 找到origin_question在truth下的位置
        index = truth.find(origin_question)

        # 开始下标
        start_index = len(context1) + len(context2)
        # 已经填充了的下标
        processed_index = []
        for idx in range(len(origin_question)):
            decoder_output[index] = start_index
            processed_index.append(index)
            start_index += 1
            index += 1

        # 剩余的下标在decoder_input去找
        for idx, decoder_indx in enumerate(decoder_output):
            if decoder_indx == -1:
                char = truth[idx]
                find_index = encoder_input.find(char)
                decoder_output[idx] = find_index

        # 0 pad 1 起始符 2 结束符

        return [idx + 2 for idx in decoder_output]

    def get_pos_vector(self, inputs):
        # 生成pos向量

        return [idx for idx in range(len(inputs))], len(inputs)

    def train_batcher(self):

        while True:
            encoder_inputs, encoder_part_inputs, decoder_inputs, decoder_outputs, encoder_pos_inputs, decoder_pos_inputs = [], [], [], [], [], []
            for idx in self.train_index:
                _context1, _context2, _uttrance, _truth = self.datas[idx]

                # decoder输出得答案
                decoder_output_idx = self.get_decoder_output(_context1, _context2, _uttrance, _truth)

                context1 = [self.char2id.get(char, 1) for char in _context1]
                context2 = [self.char2id.get(char, 1) for char in _context2]
                uttrance = [self.char2id.get(char, 1) for char in _uttrance]

                input_part1_encoding = [1] * len(context1)
                input_part2_encoding = [2] * len(context2)
                input_part3_encoding = [3] * len(uttrance)

                encoder_input = context1 + context2 + uttrance + [2]  # encoder_input 末尾加入eos
                encoder_part_input = input_part1_encoding + input_part2_encoding + input_part3_encoding + [0]

                # decoder 的入 移除最后个一token，并在第将第一个token设置为start
                # decoder 的 输出 最后一个token补冲一个end

                # decoder 1 起始符 2结束符 0 pad
                decoder_input = [1] + decoder_output_idx  # -1起始符
                decoder_output = decoder_output_idx + [len(encoder_input) - 1]  # 最后一个位置输出EOS

                # 生成pos向量
                encoder_pos, encoder_maxlen = self.get_pos_vector(encoder_input)
                decoder_pos, decoder_maxlen = self.get_pos_vector(decoder_input)

                encoder_inputs.append(encoder_input)
                encoder_part_inputs.append(encoder_part_input)
                decoder_inputs.append(decoder_input)
                decoder_outputs.append(decoder_output)
                encoder_pos_inputs.append(encoder_pos)
                decoder_pos_inputs.append(decoder_pos)

                if len(encoder_inputs) == self.batch_size or idx == self.train_index[-1]:
                    encoder_inputs = np.array(self.seq_padding(encoder_inputs))
                    encoder_part_inputs = np.array(self.seq_padding(encoder_part_inputs))
                    decoder_inputs = np.array(self.seq_padding(decoder_inputs))
                    decoder_outputs = np.array(self.seq_padding(decoder_outputs))
                    encoder_pos_inputs = np.array(self.seq_padding(encoder_pos_inputs))
                    decoder_pos_inputs = np.array(self.seq_padding(decoder_pos_inputs))

                    # 转换为cuda tensor
                    encoder_input_tensor = Variable(torch.from_numpy(encoder_inputs))
                    encoder_part_inputs_tensor = Variable(torch.from_numpy(encoder_part_inputs))
                    decoder_input_tensor = Variable(torch.from_numpy(decoder_inputs))
                    decoder_ouput_tensor = Variable(torch.from_numpy(decoder_outputs))
                    encoder_pos_tensor = Variable(torch.from_numpy(encoder_pos_inputs))
                    decoder_pos_tensor = Variable(torch.from_numpy(decoder_pos_inputs))

                    yield encoder_input_tensor, encoder_part_inputs_tensor, decoder_input_tensor, \
                          decoder_ouput_tensor, encoder_pos_tensor, decoder_pos_tensor
                    encoder_inputs, encoder_part_inputs, decoder_inputs, \
                    decoder_outputs, encoder_pos_inputs, decoder_pos_inputs = [], [], [], [], [], []


if __name__ == "__main__":
    G = data_generator('/kaggle/input/corpus2/corpus.txt', 20)
    a = G.train_batcher()
    for data in a:
        print(data)

