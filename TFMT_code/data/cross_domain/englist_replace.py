from transformers import AutoTokenizer
from nltk.corpus import stopwords  # 引入停用词，因为对停用词进行数据增强相当于没有增强
from nltk.corpus import wordnet as wn  # 引入同义词
import random

stop_words = stopwords.words('english')
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
    stop_words.append(w)


# 这里传入的words是一个列表,
# eg:"hello world".split(" ") or ["hello","world"]
def synonym_replacement(words, n, tokenizer):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word, tokenizer)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return " ".join(new_words)


def get_synonyms(word, tokenizer):
    l1 = len(tokenizer(word, add_special_tokens=False)['input_ids'])
    nearbyWordSet = wn.synsets(word)
    newSet = []
    for i in nearbyWordSet:
        for j in i.lemma_names():
            if len(tokenizer(j, add_special_tokens=False)['input_ids']) == l1 and j != word:
                newSet.append(j)
    if newSet == []:
        return [word]

    return newSet#newSet[0].lemma_names()


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

# words="hello world"
# lenW=len(words)/5#随机替换1/4的词语
# newWrods=synonym_replacement(words.split(" "),lenW, tokenizer)
# print(newWrods)

with open('15res/train_replace.txt', 'a', encoding='utf-8') as f1:
    with open('15res/train.txt', 'r', encoding='utf-8') as f2:
        words = f2.readline()
        while(words):
            # print(words)
            words, x = words.split('####')
            lenW = len(words.split(" ")) / 4
            newWrods = synonym_replacement(words.split(" "), lenW, tokenizer)
            newWrods = newWrods + '####' + x
            f1.write(newWrods)
            words = f2.readline()


