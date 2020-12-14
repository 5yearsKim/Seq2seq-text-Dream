import random
from torch import nn
import torch
from torch import Tensor


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5,
                is_dream: bool = False) -> Tensor:
        
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(src.device)
        attns = []

        encoder_outputs, hidden = self.encoder(src, is_dream)

        # first input to the decoder is the <sos> token
        output = trg[0, :]
        
        for t in range(1, max_len):
            output, hidden, a = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            attns.append(a.squeeze())
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs, torch.stack(attns)