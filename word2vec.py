import logging

from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
from pandas import read_csv
from sklearn import cluster
from sklearn.externals import joblib


class LemmTextIterator:
    def __init__(self, stop_words, filename='out.csv'):
        self.data = read_csv(filename)
        self.stop_words = stop_words

    def __iter__(self):
        for i, row in self.data.iterrows():
            doc = [word for word in row['lemm_text'].split() if word not in self.stop_words]
            yield doc


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


def main():
    # Тренировка модели, можно натренировать 1 раз и закомментить
    # train_model()

    # Загрузка векторов слов модели, натренированной train_model()
    wv = load_word_vectors()

    # Тренировка kmeans на векторах слов, можно 1 раз натренироваться, а потом только грузить
    # train_kmeans(wv)

    # Загрузка предтренированной модели kmeans
    loaded_kmeans = load_kmeans()

    write_clusters_to_file(loaded_kmeans, wv)


if __name__ == '__main__':
    main()
