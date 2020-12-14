import os
''' dataset cofnig '''
LOAD_FROM_DUMP = True

DATA_ROOT = './.data'

TRAIN_PATH = os.path.join(DATA_ROOT, 'train.pkl')
VAL_PATH = os.path.join(DATA_ROOT, 'val.pkl')
TEST_PATH = os.path.join(DATA_ROOT, 'test.pkl')
DH_PATH = os.path.join(DATA_ROOT, 'data_handler.pkl')

''' model config '''
INPUT_DIM = 19214 # len(de_vocab)
OUTPUT_DIM = 10840 # len(en_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

''' training config '''
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 10
PRINT_FREQ = 500
VAL_BEST_PATH = f'ckpts/best.pt'

""" dream config """
MAX_LEN = 8
ENTROPY_S = .1
DREAM_EPOCHS = 100000
DREAM_LR = 0.01
DREAM_PRINT_FREQ = 250
DREAM_VAL_FREQ = 500

