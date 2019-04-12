from easydict import EasyDict as edict
import os
import os.path as osp

# Supported characters
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'_&.!?,\""
# CHAR_VECTOR = "0123456789"

cfg = edict()
# Number of classes
cfg.NUM_CLASSES = len(CHAR_VECTOR) + 2
cfg.SEQ_LENGTH = 25
cfg.CHAR_VECTOR = CHAR_VECTOR
cfg.IMAGE_SHAPE = (32, 100, 1)
cfg.NUM_UNITS = 256

cfg.PATH = edict()
cfg.PATH.ROOT_DIR = os.getcwd()
cfg.PATH.TBOARD_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'logs'))
cfg.PATH.MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'checkpoints'))

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.LEARNING_RATE = 0.0001
cfg.TRAIN.LR_DECAY_STEPS = 10000
cfg.TRAIN.LR_DECAY_RATE = 0.9
cfg.TRAIN.EPOCHS = 50000
cfg.TRAIN.DISPLAY_STEP = 100
cfg.TRAIN.GPU_MEMORY_FRACTION = 0.5
cfg.TRAIN.TF_ALLOW_GROWTH = True

# VALID
cfg.VALID = edict()
cfg.VALID.BATCH_SIZE = 4
