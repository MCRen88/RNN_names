import glob
import unicodedata
import string
import torch

all_text_files = glob.glob ('data/names/*.txt')
print (all_text_files)

# В данный момент, названия находятся в формате Unicode.
# Однако, нам нужно конвертировать их в стандарт ASCII. 
# Это поможет с удалением диакритиков в словах.
# Например, французское имя Béringer будет конвертировано в Beringer

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
            )
print (unicode_to_ascii('Béringer'))

# Следующий шаг — создание словаря со списком имен для каждого языка
category_languages = {}
all_categories = []

def readLines (filename):
    lines = open (filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in all_text_files:
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    languages = readLines(filename)
    category_languages[category] = languages
    
no_of_languages = len(all_categories)
print('There are {} languages'.format(no_of_languages))

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
print (letter_to_tensor('M'))