import nltk
from nltk import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from pandas import read_csv


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# data['text'].apply(lambda x: ' '.join(x.split()))
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
data = read_csv('songdata.csv')
lemmatizer = WordNetLemmatizer()
lemmatized_texts = []

for i, row in data.iterrows():
    # Чтобы не впасть в пучины отчаяния, наблюдая за тем, как лемматизация идёт 2.5 часа
    print(i)
    text = row['text']
    tokenizer = RegexpTokenizer(r'\w+')
    # Конвертируем строку в нижний регистр и разбиваем текст песни на слова
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w
              in tokenizer.tokenize(text.lower())]

    # отбрасываем числа, если они были в тексте и слова длины 1
    tokens = [token for token in tokens if
              (not token.isnumeric() and len(token) > 1)]
    lemmatized_texts.append(' '.join(tokens))

data['lemm_text'] = lemmatized_texts
data.to_csv('out.csv', encoding='utf-8')
