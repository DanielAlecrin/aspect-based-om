import train_nb_model, train_svm_model
import pandas as pd
import pre_processing
import re

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
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

def main():
        with open('data/data_train.csv', 'r', encoding='utf8') as csvfile:
                review_list = pd.read_csv(csvfile)

        train_clean_sentences = []
        for text in review_list['text']:
                text = text.strip()
                cleaned = clean_sentence(text)
                cleaned = ' '.join(cleaned)
                train_clean_sentences.append(cleaned)

        sentences = pd.Series(train_clean_sentences)
        polarities = review_list['polarity']

        print('#####################   NAIVE BAYES   #########################')
        print()
        probs_nb = train_nb_model.train_model(sentences, polarities)
        print()
        print('#####################   SVM   #########################')
        print()
        probs_svm = train_svm_model.train_model(sentences, polarities)

        generate_roc_curve(polarities, probs_nb, probs_svm)

def generate_roc_curve(polarities, probs_nb, probs_svm):
    #auc = roc_auc_score(polarities, probs)
    #print('AUC: %.2f' % auc) 

    fpr_nb, tpr_nb, thresholds_nb = roc_curve(polarities, probs_nb)
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(polarities, probs_svm)

    colors = ['blue', 'green']
    lw = 2
    #for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_nb, tpr_nb, color=colors[0], lw=lw,
            label='NB')
    plt.plot(fpr_svm, tpr_svm, color=colors[1], lw=lw,
            label='SVM')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()