import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
import io


UKN_TKN = 0
PAD_TKN = 1
BOS_TKN = 2
EOS_TKN = 3


class DataHandler:
    def __init__(self, vocab_path):
        self.de_tokenizer = get_tokenizer('spacy', language='de')
        self.en_tokenizer = get_tokenizer('spacy', language='en')

        self.de_vocab = self.build_vocab(vocab_path[0], self.de_tokenizer)
        self.en_vocab = self.build_vocab(vocab_path[1], self.en_tokenizer)
        print(f'de_vocab len : {len(self.de_vocab)}, en_vocab len : {len(self.en_vocab)}')

    def build_vocab(self, filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    def data_process(self, filepaths):
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            de_indexes = [self.de_vocab[token] for token in self.de_tokenizer(raw_de)]
            en_indexes = [self.en_vocab[token] for token in self.en_tokenizer(raw_en)]
            data.append((de_indexes, en_indexes))
        return data

def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.tensor([BOS_TKN, *de_item, EOS_TKN], dtype=torch.long))
        en_batch.append(torch.tensor([BOS_TKN, *en_item, EOS_TKN], dtype=torch.long))
    de_batch = pad_sequence(de_batch, padding_value=PAD_TKN)
    en_batch = pad_sequence(en_batch, padding_value=PAD_TKN)
    return de_batch, en_batch

if __name__ == "__main__":
    
    # dump_data(sample, 'sample.json')
    print(load_dump('sample.json')) 