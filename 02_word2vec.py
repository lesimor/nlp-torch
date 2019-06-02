# Code by ByungWook.Kang @lesimor
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib

dtype = torch.FloatTensor

sentences = [
    "전신은 두 갈래로 나뉘는데 전자는 1951년 구진현이 세운 재단법인 '훈세사'를 모태로 하여 발기인 11명에 의해 '한국안보화재해상재보험'으로 창립하였고 1955년 대한잠사회가 인수하였다. 후자는 1956년 장기영, 박흥식 등이 안국화재를 설립했으나 1958년 삼성그룹에 인수되었고 1962년 안보화재에 합병된 후 1963년 안국화재해상보험으로 재출범하였다. 1993년 현재 사명인 삼성화재해상보험으로 바뀌었다. 2016년 을지로 사옥을 부영그룹에 매각하고 다른 삼성 금융사들과 함께 서초동 사옥으로 이전하였다."]

word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}

# Hyper parameters
batch_size = 20
embedding_size = 2
window_size = 2
voc_size = len(word_list)


def find_index(w):
    return word_dict.get(w, -1)


skip_gram = []
for x in range(window_size, len(word_sequence) - window_size):
    for w in range(1, window_size + 1):
        target_idx = find_index(word_sequence[x])

        # Convert target to project row of hidden layer
        target = np.eye(voc_size)[target_idx]

        skip_gram.append([target, find_index(word_sequence[x + w])])
        skip_gram.append([target, find_index(word_sequence[x - w])])



class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()

        self.W = nn.Parameter(torch.rand(voc_size, embedding_size)).type(dtype)
        self.WT = nn.Parameter(torch.rand(embedding_size, voc_size)).type(dtype)

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = torch.matmul(X, self.W)  # hidden: [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.WT)  # output: [batch_size, voc_size]
        return output_layer


model = Word2Vec()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10000):
    sampled = random.sample(skip_gram, batch_size)

    input_batch = Variable(torch.Tensor([x[0] for x in sampled]))
    target_batch = Variable(torch.LongTensor([x[1] for x in sampled]))

    optimizer.zero_grad()

    output = model(input_batch)
    loss = criterion(output, target_batch)

    if epoch % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

matplotlib.rc('font', family='NanumMyeongjoOTF')
for i, label in enumerate(word_list):
    W, WT = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x,y), xytext=(5,2),textcoords='offset points', ha='right', va='bottom')

plt.show()