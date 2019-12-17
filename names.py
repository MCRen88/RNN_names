import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random

import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#all_text_files = glob.glob ('data/names/*.txt')
#print (all_text_files)

# В данный момент, названия находятся в формате Unicode.
# Однако, нам нужно конвертировать их в стандарт ASCII. 
# Это поможет с удалением диакритиков в словах.
# Например, французское имя Béringer будет конвертировано в Beringer
criterion = nn.NLLLoss()
learning_rate = 0.005

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
            )
#print (unicode_to_ascii('Béringer'))

# Следующий шаг — создание словаря со списком имен для каждого языка
category_languages = {}
all_categories = []

def readLines (filename):
    lines = open (filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


#print('There are {} languages'.format(no_of_languages))

#print (category_languages['Ukraine'][:15])

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate (line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor
#print (letter_to_tensor('M'))

def category_from_output(output):
    top_n, top_i = output.data.tork(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_pair():
    category = random_choice(all_categories)
    line = random_choice(category_languages[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

###############################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
 
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
 
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.dim = output
        return output, hidden
 
    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

###############################################################################
def train (category_tensor, line_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()
        
    return output, loss.data[0]

n_epochs = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = random_training_pair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print ('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))
        
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        
plt.figure()
plt.plot(all_losses)

no_of_languages = len(all_categories)
n_hidden = 128
rnn = RNN (n_letters, n_hidden, no_of_languages)
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)

confusion = torch.zeros(no_of_languages, no_of_languages)
n_confusion = 10000

def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_pair()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1
    
for i in range(no_of_languages):
    confusion[i] = confusion[i] / confusion[i].sum()
    
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

###############################################################################
#predict
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    output = evaluate(Variable(line_to_tensor(input_line)))
 
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []
 
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])
 
predict('Austin')
###############################################################################
if __name__ == '__main__':
    
    category_languages = {}
    all_categories = []
    
    all_text_files = glob.glob('data/names/*.txt')
    
    for filename in all_text_files:
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        languages = [unicode_to_ascii(line.strip()) for line in open(filename).readlines()]
        category_languages[category] = languages

###############################################################################
# Тренировка RNN
    for i in range(10):
       category, line, category_tensor, line_tensor = random_training_pair()
       print('category= ', category, '/ line', line)
        
 
    #input = Variable(letter_to_tensor('D'))
    #hidden = rnn.init_hidden()
    
    #output, next_hidden = rnn(input, hidden)
    #print('output.size= ', output.size())
    
    #input = Variable(line_to_tensor('Derrick'))
    #hidden = Variable(torch.zeros(1, n_hidden))
    
    #output, next_hidden = rnn(input[0], hidden)
    #print (output)