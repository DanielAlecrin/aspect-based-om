import re
import string
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def review_cleanup_symbols(review_sentences):
    """
    Pré-processamento: Limpa caracteres indesejados através de expressão regular
    """
    reg_exp_main = re.compile('[^A-Za-z0-9^\n^\.^\"^\'^\- ]+', re.IGNORECASE | re.DOTALL)
    review_filtered_main = re.sub(reg_exp_main, '', review_sentences)
    return review_filtered_main

def lemmatization_sentence(sentence):
    """
    Lemmatization: Retira as variações das palavras
    """
    sentence_after_lemmatization = []
    lemmatizer = WordNetLemmatizer()
    for words in sentence.split():
        lemma_word = lemmatizer.lemmatize(words)
        sentence_after_lemmatization.append(lemma_word)
    return ' '.join(sentence_after_lemmatization)


def filter_stopwords(candidate_aspect_list):
    """
    Filtra as stopwords - (English)
    """
    stop_words = set(stopwords.words('english'))
    new_list = ('(', ')', '.', '-', '--', '``', "'", '"', "ha", "wa", "lot")
    stop_words.update(new_list)
    aspect_list_without_stopwords = []
    for item in candidate_aspect_list:
        review_id = item[0]
        words = item[1]
        product_aspect = []
        for w in words:
            if w not in stop_words and w != '' and len(w) > 1:
                product_aspect.append(w)
        if product_aspect:
            aspect_per_sent_after_stopwords = (review_id, product_aspect)
            aspect_list_without_stopwords.append(aspect_per_sent_after_stopwords)
    return aspect_list_without_stopwords

def removeNumbers(item):
    res = ''.join([i for i in item if not i.isdigit()])
    item = res
    return item

def removePunctuationsExpressions(item):
    exclude = set(string.punctuation)
    item = ''.join(ch for ch in item if ch not in exclude)
    return item

def preprocessing(reviewList):
    identifiers = reviewList['reviewId']
    text = reviewList['text']

    text = text.str.lower()
    listOfSentences = list()
    contractions = pd.read_json('assets/contractions.json', typ='series')
    for m in range(len(list(text))):
        item = removeNumbers(text[m])
        item = removePunctuationsExpressions(item)
        # item = posTagging(item)
        contractred = item.split()
        new_text = []
        for word in contractred:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        item = " ".join(new_text)

        listOfSentences.append({ 'review_id': identifiers[m], 'sentence': item})

    return listOfSentences