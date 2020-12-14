from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
    
    def embed_from_prob(self, prob):
        assert prob.shape[-1] == self.weight.shape[0]
        return torch.matmul(prob, self.weight)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = CustomEmbedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor,
                is_dream : bool) -> Tuple[Tensor]:
        if is_dream:
            embedded = self.embedding.embed_from_prob(src)
        else:
            embedded = self.embedding(src)
        if self.dropout != 0.:
            embedded = self.dropout(embedded)

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


