import os

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE_NAME = 'train.txt'
TRAIN_FILE_PATH = os.path.join(SOURCE_DIR, TRAIN_FILE_NAME)
if not os.path.isfile(TRAIN_FILE_PATH):
    f = open(TRAIN_FILE_PATH, 'w')
    f.write('__label__1 i love you\n')
    f.write('__label__1 he loves me\n')
    f.write('__label__1 she likes baseball\n')
    f.write('__label__0 i hate you\n')
    f.write('__label__0 sorry for that\n')
    f.write('__label__0 this is awful')
    f.close()

TEST_FILE_NAME = 'test.txt'
TEST_FILE_PATH = os.path.join(SOURCE_DIR, TEST_FILE_NAME)
if not os.path.isfile(TEST_FILE_PATH):
    f = open(TEST_FILE_PATH, 'w')
    f.write('sorry hate you')
    f.close()