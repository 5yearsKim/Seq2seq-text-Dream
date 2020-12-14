import torch
from dataloader import prepare_data
from model import Encoder, Attention, Decoder, Seq2Seq, init_weights
from inferencer import Inferencer
from config import *

""" load data """
train_loader, val_loader, test_loader, m_dh = prepare_data(TRAIN_PATH, VAL_PATH, TEST_PATH, DH_PATH, LOAD_FROM_DUMP, 3) 

""" model setup """
INPUT_DIM, OUTPUT_DIM = len(m_dh.de_vocab), len(m_dh.en_vocab)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec)

""" load model """
state_dict = torch.load('ckpts/best.pt')
model.load_state_dict(state_dict['model_state'])
model.eval()

en_infer = Inferencer(m_dh.en_vocab)

src, trg = next(iter(test_loader))

""" ______________ """
import matplotlib.pyplot as plt
import numpy

def plot_head_map(mma, target_labels, source_labels):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False) # mma.shape[1] = target seq 길이
    ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False) # mma.shape[0] = input seq 길이
 
    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))
 
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
 
    # source words -> column labels
    # ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    # ax.set_yticklabels(target_labels, minor=False)
 
    plt.xticks(rotation=45)
 
    # plt.tight_layout()
    plt.show()

print(len(src), len(trg))
guess, attn = model(src, trg, teacher_forcing_ratio=0)
guess = guess.max(2)[1]
idx = 1
vis_attn = attn[:,idx,:].T.detach().numpy()
vis_trg, vis_guess = trg[:, idx], guess[:, idx]
plot_head_map(vis_attn, vis_trg, vis_guess)
print(vis_attn.shape, vis_trg.shape, vis_guess.shape)
print(attn[:,0,:].shape)
# print(en_infer.decode(guess))
# print(en_infer.decode(trg))

