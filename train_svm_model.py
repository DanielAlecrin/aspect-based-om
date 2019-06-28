import pandas as pd
import numpy as np
import string
import re
import pickle
import time
from datetime import datetime
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import pre_processing

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
from nltk.stem.wordnet import WordNetLemmatizer

np.random.seed(500)
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
contractions = pd.read_json('assets/contractions.json', typ='series')

# Limpa as sentenças  
def clean_sentence(doc):
    cleaned = pre_processing.review_cleanup_symbols(doc)
    contractred = cleaned.split()
    new_text = []
    for word in contractred:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    cleaned = " ".join(new_text)
    stop_free = " ".join([i for i in cleaned.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

# Função responsável por treinar o classificador SVM
# Parâmetros:
    # sentences: Lista de sentenças pré-processadas
    # polarities: Lista com as respectivas polaridades de cada sentença
def train_model_svm(sentences, polarities):

    # Geração da matriz termo-documento
    tfidf_vect = TfidfVectorizer(min_df = 2, max_df = 0.5, sublinear_tf = True, max_features=5000)
    vect_data = tfidf_vect.fit_transform(sentences)

    # Algoritmo - SVM
    svm_model = SVC(kernel='linear')
    
    # Executa Cross Validation(k = 10) e retorna o score de cada test fold
    pred_svm = cross_val_score(svm_model, vect_data, polarities, cv=10)
    print("Lista de scores: ", cross_val_score(svm_model, vect_data, polarities, cv=10))
    print("Média dos scores: ", pred_svm.mean()*100)


    series = pd.Series(pred_svm)
    print("MÉDIA: ", series.mean() * 100)
    print("DESVIO PADRÃO: ", series.std() * 100)

    y_pred = cross_val_predict(svm_model, vect_data, polarities, cv=10)
    conf_mat = confusion_matrix(polarities, y_pred)
    print("Confusion matrix: ", conf_mat)

    # Transform to df for easier plotting
    ''' cm_df = pd.DataFrame(conf_mat,
                        index = ['Negativo','Positivo'], 
                        columns = ['Negativo','Positivo'])

    plt.figure(figsize=(10,7))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    #plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(polarities, y_pred)))
    plt.title('SVM Linear Kernel \nAcurácia: 79,5%')
    plt.ylabel('Classe correta')
    plt.xlabel('Classe prevista')
    plt.show()
 '''
    print()
    print()
    print(np.mean(polarities == y_pred))
    
    print(precision_recall_fscore_support(polarities, y_pred, labels=[-1, 1]))
    print("\n Classification Report \n", classification_report(y_pred, polarities))
    print("\n Accuracy : ", accuracy_score(polarities, y_pred))

    # pickling the model and vectorizer
    pickle.dump(pred_svm, open('models/svm_classifier.sav', 'wb'))
    pickle.dump(tfidf_vect, open('models/svm_vectorizer.sav', 'wb'))

def main():
    with open('data/data_train.csv', 'r', encoding='utf8') as csvfile:
        review_list_origin = pd.read_csv(csvfile)
    
    start_time = datetime.now()

    teste = review_list_origin.loc[review_list_origin['polarity'] == 1]
    teste2 = review_list_origin.loc[review_list_origin['polarity'] == -1]
    review_list = pd.concat([teste, teste2])

    train_clean_sentences = []
    for text in review_list['text']:
        text = text.strip()
        cleaned = clean_sentence(text)
        cleaned = ' '.join(cleaned)
        train_clean_sentences.append(cleaned)

    sentences = pd.Series(train_clean_sentences)

    train_model_svm(sentences, review_list['polarity'])


if __name__ == '__main__':
    main()