import time

import gensim
import math
import numpy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.test.utils import datapath
from pandas import read_csv
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Вероятности для введенного текста (уже лемматизирован) - да, это копипаста с соседнего файла<3
def create_probabilities(text):
    temp_file = datapath("model")
    lda = LdaModel.load(temp_file)
    dictionary = Dictionary.load('dictionary')
    doc2bow = dictionary.doc2bow(text.split())
    topic_probabilities = lda[doc2bow]
    probabilities = [t[1] for t in topic_probabilities]
    return (probabilities)

def euclid_normalized(v):
    return v / numpy.linalg.norm(v)

def get_recommendations(text):
    normalized_all = []
    song_vector = [create_probabilities(text)]
    my_data = numpy.genfromtxt('new_data_set.csv', delimiter=',')
    for vec in my_data:
        normalized_all.append(preprocessing.normalize([vec], axis=1))
    normalized = preprocessing.normalize(song_vector)
    cosine = []
    for norm_vector in normalized_all:
        cosine.append(numpy.dot(norm_vector[0], normalized[0]))
    data = read_csv('out.csv')
    indices = numpy.asarray(cosine).ravel().argsort()
    with open('box_recommendations.txt', mode='w+') as output:
        for index in sorted(indices[-10:], reverse=True):
            output.write(f"{cosine[index]}\n{data[['artist','song']].iloc[index].to_string()}\n\n" )

def multiply_scalar(v1,v2):
    if len(v1)!=len(v2):
        raise Exception("Vectors have different lengths")
    res = 0
    for i in range(len(v1)):
        res += v1[i]*v2[i]
    return res

def vanila_recommendations(text):
    mysong = create_probabilities(text)
    inv_length = 1 / math.sqrt(sum([i ** 2 for i in mysong]))
    mysong_normalized = [x * inv_length  for x in mysong]
    my_data = numpy.genfromtxt('new_data_set.csv', delimiter=',')
    my_data_normalized = []
    cosines = []
    for vec in my_data:
        inv_length = 1 / math.sqrt(sum([i ** 2 for i in vec]))
        vec_normalized = vec * inv_length
        my_data_normalized.append(vec_normalized)
        cosines.append(multiply_scalar(mysong_normalized, vec_normalized))
    data = read_csv('out.csv')
    indices = sorted(range(len(cosines)), key=lambda i: cosines[i], reverse=True)[:10]
    with open('vanila_recommendations.txt', mode='w+') as output:
        for index in indices:
            output.write(f"{cosines[index]}\n{data[['artist', 'song']].iloc[index].to_string()}\n")


def andrey_vanila_recommendations(text):
    mysong = create_probabilities(text)
    mysong_normalized = euclid_normalized(mysong)
    my_data = numpy.genfromtxt('new_data_set.csv', delimiter=',')
    my_data_normalized = []
    cosines = []
    for vec in my_data:
        vec_normalized = euclid_normalized(vec)
        my_data_normalized.append(vec_normalized)
        cosines.append(multiply_scalar(mysong_normalized, vec_normalized))
    data = read_csv('out.csv')
    indices = sorted(range(len(cosines)), key=lambda i: cosines[i], reverse=True)[:10]
    with open('euclid_normalized.txt', mode='w+') as output:
        for index in indices:
            output.write(f"{cosines[index]}\n{data[['artist', 'song']].iloc[index].to_string()}\n")

text = input("Введите что-нибудь: ")

start_time = time.time()
get_recommendations(text)
print(f"box realisation: {time.time() - start_time} seconds" )

start_time = time.time()
vanila_recommendations(text)
print(f"vanila realisation: {time.time() - start_time} seconds")

start_time = time.time()
andrey_vanila_recommendations(text)
print(f"euclead normalization realisation: {time.time() - start_time} seconds")