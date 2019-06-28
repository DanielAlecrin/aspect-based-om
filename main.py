import aspects_extraction,\
    pos_tagging, aspect_pruning, asr_apriori, summary
import pre_processing
import pandas as pd
import numpy as np
import database

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc  
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize, LabelBinarizer  

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

from datetime import datetime
import functools
import matplotlib.pyplot as plt
import seaborn as sns

import string
import re
import pickle
import time

np.random.seed(500)

def main():
    with open('data/data_test.csv', 'r', encoding='utf8') as csvfile:
        review_list_origin = pd.read_csv(csvfile)
    
    teste = review_list_origin.loc[review_list_origin['polarity'] == 1]
    teste2 = review_list_origin.loc[review_list_origin['polarity'] == -1]
    review_list = pd.concat([teste, teste2], ignore_index=True)
    start_time = datetime.now()

    """Extrai cada sentença do comentário, pré processa e armazena no banco"""
    sentence_prepocessed_list = pre_processing.preprocessing(review_list)
    sentence_list_saved = database.fetch_sentences_from_review(sentence_prepocessed_list)

    """Stanford POS tagging e armazenamento no banco"""
    pos_tagging.stanford_pos_tagging(sentence_list_saved)
    """Extraindo substantivos (Nouns), frase nominais (Noun phrases) e armazenando no banco"""
    aspects_extraction.extract_nouns_from_stanford_pos()

    """Algoritmo Apriori para conjuntos de items frequentes"""
    asr_apriori.asr_apriori_itemset()
    asr_apriori.frequent_itemset_from_db()

    """Verifica aspectos com 2 ou mais palavras e remove as que não possuem significado, ou seja, 
       que não aparecem várias vezes na mesma ordem"""
    aspect_pruning.remove_aspects()
    candidate_product_aspect = database.fetch_features_after_remove_aspects()
    database.insert_final_product_aspect_list(candidate_product_aspect)

    """Calcula o tempo total"""
    end_time = datetime.now()
    execution_time = end_time - start_time
    print("Duração. {} ".format(execution_time))

if __name__ == '__main__':
    main()
