import csv
import logging

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.test.utils import datapath
from nltk.corpus import stopwords
from pandas import read_csv


class LemmTextIterator:
    def __init__(self, filename='out.csv'):
        self.data = read_csv(filename)

    def __iter__(self):
        for i, row in self.data.iterrows():
            yield row['lemm_text'].split()


class Doc2BowIterator:
    def __init__(self, dictio, filename='out.csv'):
        self.data = read_csv(filename)
        self.dictionary = dictio

    def __iter__(self):
        for i, row in self.data.iterrows():
            yield self.dictionary.doc2bow(row['lemm_text'].split())


def train():
    iterator = LemmTextIterator()
    dictionary = Dictionary(iterator)
    stop_words = stopwords.words('english')
    print('необработанный словарь', dictionary)
    # Отбрасываем стоп-слова
    dictionary.filter_tokens(bad_ids=(dictionary.token2id[stopword] for stopword in stop_words
                                      if stopword in dictionary.token2id))
    print('словарь без стоп-слов', dictionary)
    # Отбрасываем слишком частые и слишком редкие слова
    dictionary.filter_extremes(no_below=5)
    dictionary.filter_extremes(no_above=0.5)
    print('словарь без стоп-слов и слишком редких/частых слов', dictionary)
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.DEBUG)
    logging.root.level = logging.DEBUG

    doc2bow_iterator = Doc2BowIterator(dictionary)
    print('ПУСТЬ ОНА ТРЕНИТСЯ, Я ЕЩЁ ВТОРОЙ ПРОХОД ДОБАВИЛ.'
          'ЗАВАРИ СЕБЕ ЧАЮ, СЪЕШЬ ТОРТИК, БУЛОЧКУ, СДЕЛАЙ ПРОЕКТ.'
          'АЛГОРИТМ ТУТ САМ РАЗБЕРЁТСЯ(но нескоро)')
    lda = LdaModel(doc2bow_iterator, num_topics=50, iterations=700, alpha='auto', eta='auto',
                   id2word=dictionary, minimum_probability=0, eval_every=None, passes=2)

    # Запись в файл просто для наглядности тем
    with open('topics.txt', 'w+', encoding='utf-8') as out:
        for i in range(lda.num_topics):
            topic_terms = lda.get_topic_terms(i, topn=20)
            for term in topic_terms:
                out.write(dictionary.get(term[0]) + ' ')
            out.write('\n')

    temp_file = datapath("model")
    lda.save(temp_file)  # Сохраняем обученную модель
    dictionary.save("dictionary")  # Сохраняем словарь
    '''
    можно удалить, было нужно для тестирования
    data = read_csv('out.csv')
    print(data.iloc[0]['lemm_text'])
    lemm_text = data.iloc[0]['lemm_text']
    doc2bow = dictionary.doc2bow(lemm_text.split())
    print('DOC 2 BOW', doc2bow)
    print('LDA TOPICS', lda[doc2bow])
    '''


# Этот метод раньше назывался load_model, но теперь у него другой смысл
# Каждая строка в выходном файле - 50 вероятностей того,
# что текст песни принадлежит конкретной теме из 50
def create_probabilities_dataset():
    # Загрузка обученной модели
    temp_file = datapath("model")
    lda = LdaModel.load(temp_file)
    # Загрузка словаря для модели
    dictionary = Dictionary.load('dictionary')
    data = read_csv('out.csv')
    with open('new_data_set.csv', mode='w+', encoding='utf-8', newline='') as output:
        csv_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, row in data.iterrows():
            lemm_text = row['lemm_text']
            doc2bow = dictionary.doc2bow(lemm_text.split())
            topic_probabilities = lda[doc2bow]
            probabilities = (t[1] for t in topic_probabilities)
            csv_writer.writerow(probabilities)


def main():
    """
    Один раз прогнать train(), а потом закомментить(каждый раз "топики"
    генерируются разные), это случайный процесс, поэтому натренировать
    модель заново, но не выгрузить в csv после этого смысла не имеет
    """
    train()
    create_probabilities_dataset()


if __name__ == '__main__':
    main()
