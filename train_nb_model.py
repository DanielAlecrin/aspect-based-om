import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,\
classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict

# Função responsável por treinar o classificador NB
# Parâmetros:
    # sentences: Lista de sentenças pré-processadas
    # polarities: Lista com as respectivas polaridades de cada sentença
def train_model(sentences, polarities):

    # Geração da matriz termo-documento
    tfidf_vect = TfidfVectorizer(min_df = 2, max_df = 0.5, sublinear_tf = True, max_features=5000)
    vect_data = tfidf_vect.fit_transform(sentences)

    # Algoritmo - NB
    naive_bayes_model = MultinomialNB(alpha = 0.5)

    # Executa Cross Validation(k = 10) e retorna o score de cada test fold
    pred_nb = cross_val_score(naive_bayes_model, vect_data, polarities, cv=10)
    series = pd.Series(pred_nb)
    print("Lista de scores: ", pred_nb)
    print()
    print("Acurácia Média: ", series.mean() * 100)
    print()
    print("Desvio Padrão: ", series.std() * 100)

    y_pred = cross_val_predict(naive_bayes_model, vect_data, polarities, cv=10)
    conf_mat = confusion_matrix(polarities, y_pred, labels=[1, -1])
    print()
    print("Matriz de confusão: ", conf_mat)
    print()
    print("\n Classification Report \n", classification_report(y_pred, polarities))

    #print(np.mean(polarities == y_pred)) # Exibe acurácia média
    #print(precision_recall_fscore_support(polarities, y_pred, labels=[1, -1]))
    #print("\n Acurácia : ", accuracy_score(polarities, y_pred)) # Exibe acurácia média

    # pickling the model and vectorizer
    pickle.dump(naive_bayes_model.fit(vect_data, polarities), open('models/nb_classifier.sav', 'wb'))
    pickle.dump(tfidf_vect, open('models/nb_vectorizer.sav', 'wb'))

    # data for ROC curve
    y_pred_roc = cross_val_predict(naive_bayes_model, vect_data, polarities, cv=10, method='predict_proba')
    predicted_values_pos_class = y_pred_roc[:, 1]
    return predicted_values_pos_class
