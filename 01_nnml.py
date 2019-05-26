# Code by ByungWook.Kang @lesimor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = ['한국영화 최초로 황금종려상을 거머쥔 기생충의 봉준호 감독은 25일(현지시간) 한국 기자들과 만난 자리에서 수상소감을 이같이 밝혔다',
             '상기된 모습의 봉 감독은 수상을 예상했는지를 묻자 아뇨라고 답한 뒤 차례대로 발표하니 허들을 넘는 느낌이었다',
             '뒤로 갈수록 마음은 흥분되는데 현실감은 점점 없어졌다']

vocabs = list(set(' '.join(sentences).split()))
vocab_dict = {vocab: idx for idx, vocab in enumerate(vocabs)}
label_dict = {idx: vocab for idx, vocab in enumerate(vocabs)}

STEP_SIZE = 2
vocab_size = len(vocabs)
vocab_features = 2

n_hidden = 3

inputs = []
outputs = []

# Make batches
for s in sentences:
    splitted = s.split()
    for idx in range(len(splitted) - 2):
        inputs.append([vocab_dict[splitted[idx]], vocab_dict[splitted[idx+1]]])
        outputs.append(vocab_dict[splitted[idx + 2]])

for x, y in zip(inputs, outputs):
    print('{} -> {}', x, y)


# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(vocab_size, vocab_features)
        self.H = nn.Parameter(torch.randn(STEP_SIZE * vocab_features, n_hidden)).type(dtype)
        self.b1 = nn.Parameter(torch.randn(n_hidden)).type(dtype)
        self.U = nn.Parameter(torch.randn(n_hidden, vocab_size).type(dtype))
        self.b2 = nn.Parameter(torch.randn(vocab_size)).type(dtype)

    def forward(self, input):
        # input => [W[t-n+1] vocab idx, W[t-n+2] vocab idx, ...]
        X = self.C(input)
        X = X.view(-1, STEP_SIZE * vocab_features)
        tanh = torch.tanh(self.b1 + torch.mm(X, self.H))
        output = self.b2 + torch.mm(tanh, self.U)
        return output


model = NNLM()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(outputs))


# training
for epoch in range(0, 50000):
    optimizer.zero_grad()
    output = model(input_batch)

    loss = loss_function(output, target_batch)

    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
# print([[label_dict[input[0]], label_dict[input[1]]] for input in inputs], '->', [label_dict[n.item()] for n in predict.squeeze()])

for input, n in zip(inputs, predict.squeeze()):
    print([label_dict[input[0]], label_dict[input[1]]], '->', label_dict[n.item()])