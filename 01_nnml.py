# Code by ByungWook.Kang @lesimor
import torch.nn as nn

sentences = ['한국영화 최초로 황금종려상을 거머쥔 기생충의 봉준호 감독은 25일(현지시간) 한국 기자들과 만난 자리에서 수상소감을 이같이 밝혔다',
             '상기된 모습의 봉 감독은 수상을 예상했는지를 묻자 아뇨라고 답한 뒤 차례대로 발표하니 허들을 넘는 느낌이었다',
             '뒤로 갈수록 마음은 흥분되는데 현실감은 점점 없어졌다']

tokens = list(set(' '.join(sentences).split()))
token_dict = {token: idx for idx, token in enumerate(tokens)}
label_dict = {idx: token for idx, token in enumerate(tokens)}

WINDOW_SIZE = 2
vocab_size = len(tokens)
vocab_features = 2

inputs = []
outputs = []

# Make batches
for s in sentences:
    splitted = s.split()
    for idx in range(len(splitted) - 2):
        inputs.append([token_dict[splitted[idx]], token_dict[splitted[idx+1]]])
        outputs.append(token_dict[splitted[idx + 2]])

for x, y in zip(inputs, outputs):
    print('{} -> {}', x, y)


# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(vocab_size, 2)


