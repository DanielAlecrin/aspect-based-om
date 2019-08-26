import aspects_extraction, pos_tagging, aspect_pruning, asr_apriori, pre_processing
import pandas as pd
import database
from datetime import datetime

def main():
    with open('data/data_train.csv', 'r', encoding='utf8') as csvfile:
        review_list = pd.read_csv(csvfile)
    
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
