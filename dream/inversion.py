import torch
from torch import nn
import torch.nn.functional as F
from .utils import entropy_loss, pseudo_attn

class Inversion:
    def __init__(self, model, max_len, input_dim, criterion, entropy_s=1., inferencer=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.max_len = max_len
        self.input_dim = input_dim
        self.criterion = criterion
        self.entropy_s = entropy_s
        self.inferencer = inferencer

    def inverse(self, target, epochs, lr=0.01, print_freq=100, val_freq=200):
        bs = target.shape[1]
        target = target.to(self.device)
        logit = torch.randn((self.max_len, bs, self.input_dim), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([logit], lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            text_prob = F.softmax(logit, dim=2)
            output, attns = self.model(text_prob, target, is_dream=True, teacher_forcing_ratio=0.)
            output_flat = output[1:].view(-1, output.shape[-1])
            target_flat =  target[1:].view(-1)

            diff_loss = self.criterion(output_flat, target_flat)

            e_loss = self.entropy_s * entropy_loss(text_prob)
            
            p_attn = pseudo_attn(self.max_len, target)
            a_loss = (p_attn - attns) ** 2
            a_loss = a_loss.sum(-1).mean(-1).sum()
    
            loss = diff_loss + e_loss + a_loss
            loss.backward()

            optimizer.step()
            
            if epoch%print_freq == 0:
                print(f'epoch {epoch}: diff_loss = {diff_loss}, entropy = {e_loss.item()}, attn = {a_loss.item()}')
            if epoch%val_freq == 0:
                self.validate(text_prob, f'validate_sample/decoded_{epoch}.txt')
        print('inversion done!')

    def validate(self, text_prob, save_path="validate_sample/decoded.txt"):
        if self.inferencer is None:
            return None
        guess = self.inferencer.prob_to_seq(text_prob.detach())
        guess = self.inferencer.decode(guess)
        with open(save_path, 'w') as f:
            f.writelines([line + '\n' for line in guess])
        


