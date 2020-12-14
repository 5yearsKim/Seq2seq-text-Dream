import torch
import numpy as np

def entropy_loss(prob):
    max_len, bs, _ = prob.shape
    entropy = - prob * prob.log()
    entropy_loss = entropy.sum(-1).mean(-1).sum(-1)
    return entropy_loss

def generate_gaussian(x_len, bs, y_len):
    arr = np.arange(y_len, dtype=np.float32)
    arr = np.tile(arr, [x_len, 1])
    index = np.arange(x_len).reshape(x_len, 1) + 1 
    arr = arr - index
    gauss = np.exp(-np.square(arr))
    gauss = np.tile(np.expand_dims(gauss, axis=1), [1, bs, 1])
    return gauss

def pseudo_attn(src_len, trg):
    trg_len, bs = trg.shape[0] - 1, trg.shape[1]
    trg = trg[1:, :]
    t_filter = torch.eq(trg, 0) + torch.eq(trg, 1) 
    t_filter = (~t_filter).float().view(trg_len, bs, 1)
    gauss = torch.from_numpy(generate_gaussian(trg_len, bs, src_len)).to(torch.float) * 3
    attn = torch.nn.functional.softmax(gauss, dim =-1).to(trg.device)
    attn = attn * t_filter
    return attn
    
if __name__ == "__main__":
    import torch.nn.functional as F
    x = torch.tensor([[[1., 2., 3], [0., 1., 2]]])
    prob = F.softmax(x, dim=2)
    entropy_loss = entropy_loss(prob)
    print(prob)
    print(entropy_loss)
