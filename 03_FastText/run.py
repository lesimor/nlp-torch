import os
import fastText
import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'model')

SOURCE_DIR = os.path.join(BASE_DIR, 'src')

LABEL_MASK = '__label__'

TRAINING = True

# Train model
TRAIN_FILE_NAME = 'train.txt'
TRAIN_FILE_PATH = os.path.join(SOURCE_DIR, TRAIN_FILE_NAME)

if TRAINING:
    model = fastText.train_supervised(TRAIN_FILE_PATH, label=LABEL_MASK)
    model.save_model(MODEL_PATH)
else:
    model = fastText.load_model(MODEL_PATH)

# Test model
TEST_FILE_NAME = 'test.txt'
TEST_FILE_PATH = os.path.join(SOURCE_DIR, TEST_FILE_NAME)

test_result = model.test_label(TEST_FILE_PATH)
print(tabulate.tabulate([[k.replace(LABEL_MASK, ''), v['precision'], v['recall']] for k, v in test_result.items()],
                        headers=['Domain', 'Precision', 'Recall']))

# Predict sentence
with open(os.path.join(BASE_DIR, 'src', 'predict.txt')) as f:
    for line in f.readlines():
        text = line.strip()
        labels, predict = model.predict(text)
        print('{} -> {}'.format(text, labels[0].replace(LABEL_MASK, '')))
