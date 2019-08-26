import nltk, pre_processing
import database


def extract_nouns_from_stanford_pos():
    """
    Usa expressão regular para buscar substantivos e frases nominais(Nouns and Nouns phrases) (NN | NN| JJ-NN)
    """
    noun_list_after_chunk = []

    pos_tagged_text = database.fetach_pos_tagged_sentence()

    chunk_reg_express = r"""NP: {<JJ>*<NN.*>}"""  
    chunk_parsar = nltk.RegexpParser(chunk_reg_express)

    for review_id, pos_tagged_content in pos_tagged_text:
        
        pos_tagged_list = eval(pos_tagged_content)

        chunked = chunk_parsar.parse(pos_tagged_list)
        noun_list_per_sentence = []
        for subtree in chunked.subtrees(filter=lambda chunk_label: chunk_label.label() == 'NP'): 
            noun_list_per_sentence.append(" ".join(word for word, pos in subtree.leaves() if word not in noun_list_per_sentence))

        if noun_list_per_sentence:
            combine_value = (review_id, noun_list_per_sentence)
            noun_list_after_chunk.append(combine_value)

    # Filtra stopwords da lista de aspectos candidatos
    noun_list_without_stopwords = pre_processing.filter_stopwords(noun_list_after_chunk)
    database.insert_nouns_list_per_sentence_into_db(noun_list_without_stopwords)
    database.insert_single_candidate_aspect_per_row(noun_list_without_stopwords)
