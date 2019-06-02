# Code by ByungWook.Kang @lesimor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = ['한국영화 최초로 황금종려상을 거머쥔 기생충의 봉준호 감독은 25일(현지시간) 한국 기자들과 만난 자리에서 수상소감을 이같이 밝혔다',
             '상기된 모습의 봉 감독은 수상을 예상했는지를 묻자 아뇨라고 답한 뒤 차례대로 발표하니 허들을 넘는 느낌이었다',
             '뒤로 갈수록 마음은 흥분되는데 현실감은 점점 없어졌다',
             '금융당국은 3분기 중 인터넷은행 예비인가 절차를 다시 진행하기로 했다. 키움과 토스 모두 지적된 문제를 보완해 재도전에 나설 수 있다고 했다',
             '이지. 가장 쉬운 모드로, 초보, 루저, 하수에게 추천. 터렛 0개로도 깨기 쉬울 정도로 쉽다.노말. 적당한 모드로, 중수, 일반인에게 추천. 일반적으로, 터렛, 휠 파워를 사용해야 깰 수 있다.하드. 슬슬 어려워지기 시작하는 모드로, 고수에게 추천. 일반적으로, 스크롤, 칩셋, 터렛, 휠 파워를 모두 사용해야 깰 수 있다. 익스퍼트. 상당히 어려운 모드로, 초고수, 달인, 명인에게 추천. 일반적으로, 스크롤, 칩셋, 휠 파워, 터렛 모두 사용한 후 열심히 하며, 마지막까지 방심하지 않아야 깰 수 있다.']

vocabs = list(set(' '.join(sentences).split()))
vocab_dict = {vocab: idx for idx, vocab in enumerate(vocabs)}
label_dict = {idx: vocab for idx, vocab in enumerate(vocabs)}

STEP_SIZE = 3
vocab_size = len(vocabs)
vocab_features = 2

n_hidden = 3

inputs = []
outputs = []

# Make batches
for s in sentences:
    splitted = s.split()
    for idx in range(len(splitted) - STEP_SIZE):
        inputs.append([vocab_dict[splitted[idx + n]] for n in range(STEP_SIZE)])
        outputs.append(vocab_dict[splitted[idx + STEP_SIZE]])

for x, y in zip(inputs, outputs):
    print('{} -> {}', x, y)


# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(vocab_size, vocab_features)
        self.H = nn.Parameter(torch.randn(STEP_SIZE * vocab_features, n_hidden)).type(dtype)
        self.W = nn.Parameter(torch.randn(STEP_SIZE * vocab_features, vocab_size).type(dtype))
        self.b1 = nn.Parameter(torch.randn(n_hidden)).type(dtype)
        self.U = nn.Parameter(torch.randn(n_hidden, vocab_size).type(dtype))
        self.b2 = nn.Parameter(torch.randn(vocab_size)).type(dtype)

    def forward(self, input):
        # input => [W[t-n+1] vocab idx, W[t-n+2] vocab idx, ...]
        X = self.C(input)
        X = X.view(-1, STEP_SIZE * vocab_features)
        tanh = torch.tanh(self.b1 + torch.mm(X, self.H))
        output = self.b2 + torch.mm(X, self.W) + torch.mm(tanh, self.U)
        return output


model = NNLM()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(outputs))

# training
for epoch in range(0, 10000):
    optimizer.zero_grad()
    output = model(input_batch)

    loss = loss_function(output, target_batch)

    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
# print([[label_dict[input[0]], label_dict[input[1]]] for input in inputs], '->', [label_dict[n.item()] for n in predict.squeeze()])

for input, n in zip(inputs, predict.squeeze()):
    print([label_dict[input[n]] for n in range(STEP_SIZE)], '->', label_dict[n.item()])
