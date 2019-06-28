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

def execute_svm_roc(texts, polarities):
    Tfidf_vect = TfidfVectorizer(min_df = 2, max_df = 0.5, sublinear_tf = True, max_features=5000)
    vect_data = Tfidf_vect.fit_transform(texts)

    # Classifier - Algorithm - SVM
    pipelines_svc = SVC(kernel='linear', degree=3, gamma='auto', probability=True) #.fit(train_vectors,y_treino)
    # predict the labels on validation dataset
    pred_svc = cross_val_score(pipelines_svc, vect_data, polarities, cv=10)

    series = pd.Series(pred_svc)
    y_pred = cross_val_predict(pipelines_svc, vect_data, polarities, cv=10,method='predict_proba')

    probs = y_pred[:, 1]  
    auc = roc_auc_score(polarities, probs)  
    print('AUC: %.2f' % auc) 
    return probs, auc

# Função responsável por treinar o classificador NB
# Parâmetros:
    # sentences: Lista de sentenças pré-processadas
    # polarities: Lista com as respectivas polaridades de cada sentença
def train_model_nb(sentences, polarities):

    # Geração da matriz termo-documento
    tfidf_vect = TfidfVectorizer(min_df = 2, max_df = 0.5, sublinear_tf = True, max_features=5000)
    vect_data = tfidf_vect.fit_transform(sentences)

    # Algoritmo - NB
    naive_bayes_model = MultinomialNB(alpha = 0.5)

    # Executa Cross Validation(k = 10) e retorna o score de cada test fold
    pred_nb = cross_val_score(naive_bayes_model, vect_data, polarities, cv=10)
    print("Lista de scores: ", cross_val_score(naive_bayes_model, vect_data, polarities, cv=10))
    print("Média dos scores: ", pred_nb.mean()*100)

    series = pd.Series(pred_nb)
    print("MÉDIA: ", series.mean() * 100)
    print("DESVIO PADRÃO: ", series.std() * 100)

    y_pred = cross_val_predict(naive_bayes_model, vect_data, polarities, cv=10,method='predict_proba')
    #y_pred = cross_val_predict(pipelines_mnb, vect_data, polarities, cv=10, method='predict_proba')

    ''' probs = y_pred[:, 1]  
    auc = roc_auc_score(polarities, probs)
    print('AUC: %.2f' % auc)  
    fpr, tpr, thresholds = roc_curve(polarities, probs)

    probs_2, auc2 = execute_svm_roc(sentences, polarities)
    #auc2 = roc_auc_score(polarities, probs_2)
    fpr1, tpr1, thresholds1 = roc_curve(polarities, probs_2)

    colors = ['blue', 'green']
    lw = 2
    #for i, color in zip(range(n_classes), colors):
    plt.plot(fpr, tpr, color=colors[0], lw=lw,
            label='NB')
    plt.plot(fpr1, tpr1, color=colors[1], lw=lw,
            label='SVM')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show() '''

    y_pred = cross_val_predict(naive_bayes_model, vect_data, polarities, cv=10)
    conf_mat = confusion_matrix(polarities, y_pred, labels=[1, -1])
    print("Confusion matrix: ", conf_mat)


    # Transform to df for easier plotting
    ''' cm_df = pd.DataFrame(conf_mat,
                        index = ['Negativo','Positivo'], 
                        columns = ['Negativo','Positivo'])

    plt.figure(figsize=(10,7))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    #plt.title('Naive Bayes \nAccuracy:{0:.3f}'.format(accuracy_score(polarities, y_pred)))
    plt.title('Naive Bayes \nAcurácia: 77,5%')
    plt.ylabel('Classe correta')
    plt.xlabel('Classe prevista')
    plt.show() '''

    print()
    print()
    print(np.mean(polarities == y_pred))

    print(precision_recall_fscore_support(polarities, y_pred, labels=[1, -1]))
    print("\n Classification Report \n", classification_report(y_pred, polarities))
    print("\n Accuracy : ", accuracy_score(polarities, y_pred))

    # pickling the model and vectorizer
    pickle.dump(naive_bayes_model.fit(vect_data, polarities), open('models/nb_classifier.sav', 'wb'))
    pickle.dump(tfidf_vect, open('models/nb_vectorizer.sav', 'wb'))

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

    train_model_nb(sentences, review_list['polarity'])

if __name__ == '__main__':
    main()