import torch
from dataloader import prepare_data
from model import Encoder, Attention, Decoder, Seq2Seq, init_weights
from trainer import Trainer
from config import *

""" load data """
train_loader, val_loader, test_loader, m_dh = prepare_data(TRAIN_PATH, VAL_PATH, TEST_PATH, DH_PATH, LOAD_FROM_DUMP, BATCH_SIZE) 

""" model setup """
INPUT_DIM, OUTPUT_DIM = len(m_dh.de_vocab), len(m_dh.en_vocab)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec)
model.apply(init_weights)

""" training setup """
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = torch.nn.CrossEntropyLoss(ignore_index=1)


trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, val_best_path=VAL_BEST_PATH)
trainer.load('ckpts/best.pt')

trainer.train(epochs=EPOCHS, print_feq=PRINT_FREQ)

trainer.save(f'ckpts/translator.pt')

    
