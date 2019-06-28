from copy import deepcopy
from itertools import combinations
import database

''' Apriori algorithm: https://www.digitalvidya.com/blog/apriori-algorithms-in-data-mining/ '''
def asr_apriori_itemset():
    Lk = {}
    items_in_transaction = []

    # Minimum support para o algoritmo Apriori (1% do total de sentenças)
    number_of_sentences = len(database.fetch_sentence_from_sentence_table())
    min_sup = round(0.01*number_of_sentences)

    # Busca os aspectos candidatos do BD
    transaction = database.fetch_nouns_per_sentence()
    for review_id, itm in transaction:
        
        for each_item in itm.split(','):
            items_in_transaction.append(each_item)

    # Encontra os 1-itemsets frequentes
    # Confidence é a probabilidade de ocorrer tais itens
    C1 = generate_1_itemset(items_in_transaction)
    L1 = prune(C1, min_sup)
    if L1 != '':
        database.insert_frequent_1_itemsets(L1)

    # Encontra os 2-itemsets frequentes
    C2 = generate_2_itemset(L1)
    Ct2 = scan_in_database(C2)
    Lk[2] = prune(Ct2, min_sup)
    if Lk[2] != '':
        database.insert_frequent_k_itemsets(Lk)

def generate_1_itemset(items_in_transaction):
    C_1 = {}
    for item in items_in_transaction:
        if C_1.keys() != item:
            C_1[item] = items_in_transaction.count(item)
    return C_1


def generate_2_itemset(L):
    C_2 = []
    for i in combinations(L, 2):
        C_2.append(list(i))
    return list(C_2)

def prune(candidate_aspect_list, min_sup):
    """ Apriori Pruning com base no minimum support  """
    l_k = deepcopy(candidate_aspect_list)
    for key, value in list(l_k.items()):
        if value < min_sup:
            del l_k[key]
    return l_k

def scan_in_database(Ct):
    """
    Varre o DB para verificar se o param Ct é um subset da sentença/transação
    """
    current_candidate = {}
    transaction = database.fetch_nouns_per_sentence()
    for each_Ct in Ct:
        for review_id, itm in transaction:
            item = set(itm.split(','))
            if set(each_Ct).issubset(item):
                if str(each_Ct) not in current_candidate.keys():
                    current_candidate[str(each_Ct)] = 1
                else:
                    current_candidate[str(each_Ct)] += 1
    return current_candidate


def frequent_itemset_from_db():
    """
    Busca o itemset frequente (L1, e L2) e combina em uma única lista
    """
    frequent_1_itemsets = database.fetch_frequent_itemsets()
    frequent_k_itemsets = database.fetch_frequent_k_itemsets()
    frequent_itemsets_list = []
    for freq_1_item in frequent_1_itemsets:
        if freq_1_item not in frequent_itemsets_list:
            frequent_itemsets_list.append(freq_1_item)

    for freq_k_item in frequent_k_itemsets:
        if freq_k_item not in frequent_itemsets_list:
            frequent_itemsets_list.append(freq_k_item)

    database.insert_final_candidate_aspects(frequent_itemsets_list)
