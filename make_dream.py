import torch
from dataloader import prepare_data
from model import Encoder, Attention, Decoder, Seq2Seq, init_weights
from inferencer import Inferencer
from dream import Inversion
from config import *

""" load data """
train_loader, val_loader, test_loader, m_dh = prepare_data(TRAIN_PATH, VAL_PATH, TEST_PATH, DH_PATH, LOAD_FROM_DUMP, BATCH_SIZE) 

""" model setup """
INPUT_DIM, OUTPUT_DIM = len(m_dh.de_vocab), len(m_dh.en_vocab)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, 0)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, 0, attn)

model = Seq2Seq(enc, dec)

""" load model """
state_dict = torch.load('ckpts/best.pt')
model.load_state_dict(state_dict['model_state'])

en_infer = Inferencer(m_dh.en_vocab) 
de_infer = Inferencer(m_dh.de_vocab) 

criterion = torch.nn.CrossEntropyLoss(ignore_index=1)

src, trg = next(iter(test_loader))

trg_text = en_infer.decode(trg)
with open('validate_sample/target.txt', 'w') as f:
    f.writelines(trg_text)
print(trg_text)
inversion = Inversion(model, MAX_LEN, INPUT_DIM, criterion, ENTROPY_S, inferencer=de_infer)
inversion.inverse(trg, DREAM_EPOCHS, DREAM_LR, DREAM_PRINT_FREQ, DREAM_VAL_FREQ)