import logging
import math

import numpy
import pandas
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
from pandas import read_csv
from sklearn import cluster
from sklearn.externals import joblib

from recommendations import multiply_scalar


class LemmTextIterator:
    def __init__(self, stop_words, filename='out.csv'):
        self.data = read_csv(filename)
        self.stop_words = stop_words

    def __iter__(self):
        for i, row in self.data.iterrows():
            doc = [word for word in row['lemm_text'].split() if word not in self.stop_words]
            yield doc


# Берём массив слов, находим средний вектор всех векторов слов этого массива
def song_vector(words, stop_words, wv, ):
    words = [word for word in words if word not in stop_words and word in wv.vocab]
    num_vectors = len(words)
    vector_size = len(wv[words[0]])
    mean = numpy.zeros(vector_size)

    for word in words:
        vector = wv[word]
        for i in range(vector_size):
            mean[i] += vector[i]
    for i in range(vector_size):
        mean[i] /= num_vectors
    return mean


def train_model(save_full_model=False, save_word_vectors=True):
    stop_words = stopwords.words('english')
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.DEBUG)
    logging.root.level = logging.DEBUG
    # Обучение модели
    iterator = LemmTextIterator(stop_words=stop_words)
    model = Word2Vec(iterator, size=100, window=5, min_count=4, workers=4)
    """
    Сохранение ПОЛНОЙ модели. Её можно загрузить потом и продолжить тренировать.
    В модели много информации помимо самих векторов
    """
    if save_full_model:
        # path = get_tmpfile("word2vec.model")
        model.save("word2vec.model")

    """
    Сохранение только векторов слов. Их нельзя загрузить и продолжить тренировать.
    Это просто набор векторов, но зато он занимает меньше места в памяти. Наверное,
    мы натреним модель лишь 1 раз, и нам этого будет достаточно, и тогда сейвить только
    вектора - вполне себе хороший для нас вариант
    """
    if save_word_vectors:
        path = get_tmpfile("wordvectors.kv")
        model.wv.save(path)
        wv = KeyedVectors.load(path, mmap='r')


def load_word_vectors(path="wordvectors.kv"):
    wv = KeyedVectors.load(path, mmap='r')
    return wv


# Тренируем k-means на векторах wv
def train_kmeans(wv, save=True):
    kmeans = cluster.KMeans(n_clusters=50, n_init=20)
    kmeans.fit(wv.vectors)
    # Центроиды, то есть центры кластеров
    centroids = kmeans.cluster_centers_

    print("Centroids data")
    print(centroids)

    for i in range(len(centroids)):
        # Для каждого центроида нахожу 20 наиболее похожих на него векторов слов
        similar_words_tuples = wv.similar_by_vector(centroids[i], topn=20)
        for word_score in similar_words_tuples:
            print(word_score[0], end=' ')
        print()
    if save:
        save_kmeans(kmeans)


def save_kmeans(kmeans_model, filename='model.sav'):
    joblib.dump(kmeans_model, filename)


def load_kmeans(filename='model.sav'):
    loaded_model = joblib.load(filename)
    return loaded_model


'''
По заданной k-means модели и векторному представлению слов записывает в файл по
n_words самых близких к центру кластера слов
'''


def write_clusters_to_file(kmeans_model, wv, n_words=20):
    centroids = kmeans_model.cluster_centers_
    with open("{}_clusters_top_{}.txt".format(kmeans_model.n_clusters, n_words),
              'w+', encoding='utf-8') as out:
        for i in range(len(centroids)):
            # Для каждого центроида нахожу 20 наиболее похожих на него векторов слов
            similar_words_tuples = wv.similar_by_vector(centroids[i], topn=20)
            for word_score in similar_words_tuples:
                out.write(word_score[0] + ' ')
            out.write('\n')


def update_data_set(stop_words, wv, ):
    data = read_csv('out.csv')
    song_vectors = []
    for i, row in data.iterrows():
        lemm_text = row['lemm_text']
        song_vectors.append(song_vector(lemm_text.split(), stop_words, wv))
    data['song_vector'] = song_vectors
    columns = ['artist', 'song', 'link', 'text', 'lemm_text']

    for column in columns:
        del data[column]
    data.to_csv('song_vectors.csv', encoding='utf-8')


def get_recommendations(stop_words, wv):
    song_data = read_csv('out.csv')
    words = input('Введите ключевые слова песни\n').split()
    input_vector = song_vector(words, stop_words, wv)
    inv_length = 1 / math.sqrt(sum([i ** 2 for i in input_vector]))
    input_normalized = input_vector * inv_length
    data = read_csv('song_vectors.csv')
    cosines = []
    for i, row in data.iterrows():
        lyrics_vector = numpy.array(row['song_vector'].replace('[', '')
                                    .replace(']', '').split(),
                                    dtype=numpy.float64)
        if len(lyrics_vector) != 0:
            inv_length = 1 / math.sqrt(sum([i ** 2 for i in lyrics_vector]))
        lyrics_normalized = lyrics_vector * inv_length
        cosines.append(multiply_scalar(input_normalized, lyrics_normalized))
    indices = sorted(range(len(cosines)), key=lambda i: cosines[i], reverse=True)[:10]
    pandas.set_option('display.max_colwidth', -1)
    with open('word2vec_recommendations.txt', mode='w+') as output:
        for index in indices:
            output.write(f"{cosines[index]}\n{song_data[['artist', 'song']].iloc[index].to_string()}\n")
            text = song_data['text'].iloc[index].replace(r'\n', '\n')
            output.write(f"{text}\n")
            output.write((('*' * 50) + '\n') * 3)


def main():
    # Тренировка модели, можно натренировать 1 раз и закомментить
    # train_model()

    # Загрузка векторов слов модели, натренированной train_model()
    wv = load_word_vectors()

    # Тренировка kmeans на векторах слов, можно 1 раз натренироваться, а потом только грузить
    # train_kmeans(wv)

    # Загрузка предтренированной модели kmeans
    loaded_kmeans = load_kmeans()

    stop_words = stopwords.words('english')

    # update_data_set(stop_words, wv)
    get_recommendations(stop_words, wv)
    # Запись ключевых слов кластера в файл
    # write_clusters_to_file(loaded_kmeans, wv)


if __name__ == '__main__':
    main()
