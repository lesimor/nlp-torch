import os

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE_NAME = 'train.txt'
TRAIN_FILE_PATH = os.path.join(SOURCE_DIR, TRAIN_FILE_NAME)
if not os.path.isfile(TRAIN_FILE_PATH):
    f = open(TRAIN_FILE_PATH, 'w')
    f.write('__label__positive i love you\n')
    f.write('__label__positive he loves me\n')
    f.write('__label__positive she likes baseball\n')
    f.write('__label__negative i hate you\n')
    f.write('__label__negative sorry for that\n')
    f.write('__label__negative this is awful')
    f.close()

TEST_FILE_NAME = 'test.txt'
TEST_FILE_PATH = os.path.join(SOURCE_DIR, TEST_FILE_NAME)
if not os.path.isfile(TEST_FILE_PATH):
    f = open(TEST_FILE_PATH, 'w')
    f.write('__label__negative sorry hate you')
    f.close()

PREDICT_FILE_NAME = 'predict.txt'
PREDICT_FILE_PATH = os.path.join(SOURCE_DIR, PREDICT_FILE_NAME)
if not os.path.isfile(PREDICT_FILE_PATH):
    f = open(PREDICT_FILE_PATH, 'w')
    f.write('sorry you are awful')
    f.close()