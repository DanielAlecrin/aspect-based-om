import StanfordNLPServer
import database


def stanford_pos_tagging(sentence_list):
    """
    Stanford POS Tagging e insere no banco
    param: Lista de sentenças para o POS tagging
    """
    ids_pos_value = []
    nlp_server = StanfordNLPServer.SNLPServer()
    for review_id, sentence in sentence_list:
        pos_tagged = nlp_server.pos(sentence)  
        combine_value = (review_id, pos_tagged)
        ids_pos_value.append(combine_value)

    nlp_server.close()
    
    database.insert_postagged_sent_into_db(ids_pos_value)
