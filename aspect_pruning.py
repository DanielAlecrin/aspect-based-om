import database

# Remoção de aspectos: Permanecem os aspectos em que as palavras possuem distância 
#                      menor ou igual a 2 e estão presentes em mais de 5 sentenças
def remove_aspects():
    candidate_feature_phrase = database.fetch_final_candidate_aspects()
    sentences_list = database.fetch_sentence_from_sentence_table()

    feature_list_after_processing = []
    feature_phase = []
    feature_count_in_dict = {}

    # Retorna os aspectos candidatos que são formados por duas ou mais palavras
    for feature in candidate_feature_phrase:
        word_in_noun_phrase = feature.split()
        if len(word_in_noun_phrase) > 1:
            feature_phase.append(feature)
        else:
            feature_list_after_processing.append(feature)

    # Calcula a distância entre as duas palavras em suas respectivas sentenças
    for fp in feature_phase:
        i = 0
        for review_id, sentences in sentences_list:
            word_index_dict = {}
            for fp_word in fp.split():
                for index, word in enumerate(sentences.split()):
                    if word == fp_word:
                        word_index_dict[fp_word] = index
            if (len(word_index_dict) > 2) and (len(fp.split("_")) == len(word_index_dict)):
                list_form = list(word_index_dict.values())
                previous_value = (list_form[0])
                current_value = (list_form[1])
                next_value = (list_form[2])
                if current_value - previous_value < 2 and next_value - current_value < 2:
                    i += 1
            elif len(word_index_dict) > 1 and len(fp.split()) == len(word_index_dict):
                list_form = list(word_index_dict.values())
                previous_value = (list_form[0])
                current_value = (list_form[1])
                if current_value - previous_value < 2:
                    i += 1
            else:
                i += 0

        # Conta quantas vezes o aspecto aparece
        if feature_count_in_dict.keys() != fp:
            feature_count_in_dict[fp] = i

    # Verifica se o aspecto aparece em mais de 5 sentenças
    for key, value in feature_count_in_dict.items():
        if value > 4:
            feature_list_after_processing.append(key)

    database.insert_features_after_remove_aspects(feature_list_after_processing)
