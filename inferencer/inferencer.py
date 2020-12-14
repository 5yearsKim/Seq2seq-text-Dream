import torch

class Inferencer:
    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, guess, remove_special=True):
        # guess = guess[1:] # remove sos token
        guess = guess.permute(1, 0)
        sent_list = []
        for line in guess:
            vocab_list = []
            for idx in line.tolist():
                if remove_special and idx in [1, 2, 3]:
                    continue
                vocab_list.append(self.vocab.itos[idx])
            sent_list.append(' '.join(vocab_list))
        return sent_list

    def prob_to_seq(self, prob):
        return prob.max(dim=2)[1]

