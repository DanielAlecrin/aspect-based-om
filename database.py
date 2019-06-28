import pymysql, pre_processing

try:
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='Totvs@123456',
        db='tcc')

    cursor = connection.cursor()
except pymysql.connections.Error as err:
    print(err)


def insert_into_review_table(review):
    insert_query = ("INSERT INTO review "
                    "(review_id, review_sentence)"
                    "VALUES (%s, %s)")
    cursor.execute(insert_query, review)
    connection.commit()

def insert_sentence_into_sentence_table():
    truncate_sentence_table_sql = 'TRUNCATE TABLE sentences'
    cursor.execute(truncate_sentence_table_sql)

    sql = 'SELECT * from review'
    cursor.execute(sql)
    reviews = cursor.fetchall()
    for id, review in reviews:
        for rw in review.split("\n"):
            if not rw.isspace() and rw != '':
                sentence = pre_processing.lemmatization_sentence(rw)
                insert_sentence_query(id, sentence.lower())


def insert_sentence_query(review_id, sentence):
    if sentence != ' ':
        insert_value = (review_id, sentence)
        insert_query = ("INSERT INTO sentences "
                        "(review_id, sentence)"
                        "VALUES (%s, %s)")
        cursor.execute(insert_query, insert_value)
    connection.commit()


def fetch_sentences_from_review(review):
    truncate_review_table_sql = 'TRUNCATE TABLE review'
    cursor.execute(truncate_review_table_sql)

    for rw in review:
        item = rw['sentence']
        if item != '':
            filter_symbol_rw = pre_processing.review_cleanup_symbols(item)
            insert_into_review_table((rw['review_id'], filter_symbol_rw))

    insert_sentence_into_sentence_table()
    return fetch_sentence_from_sentence_table()


def fetch_sentence_from_sentence_table():
    select_sql = 'SELECT * from sentences'
    cursor.execute(select_sql)
    return cursor.fetchall()


def insert_postagged_sent_into_db(pos_tagged_sentences):
    truncate_pos_tagged_sentences_table_sql = 'TRUNCATE TABLE pos_tagged_sentences'
    cursor.execute(truncate_pos_tagged_sentences_table_sql)

    for review_id, sent in pos_tagged_sentences:
        convert_sent_into_string = str(sent)
        insert_value = (review_id, convert_sent_into_string)
        insert_query = ("INSERT INTO pos_tagged_sentences "
                        "(review_id, pos_tagged_sentence)"
                        "VALUES (%s, %s)")
        cursor.execute(insert_query, insert_value)
    connection.commit()


def fetach_pos_tagged_sentence():
    select_sql_query = 'SELECT * From pos_tagged_sentences'
    cursor.execute(select_sql_query)
    pos_tagged_review = list(cursor)
    return pos_tagged_review


def insert_nouns_list_per_sentence_into_db(nouns_per_sent):
    trucate_table('nouns_list_per_sentence')
    for review_id, noun_set in nouns_per_sent:
        if noun_set != '':
            nouns_in_sent = ''
            index = 0
            for noun in noun_set:
                if index != len(noun_set)-1:
                    nouns_in_sent += noun + ','
                    index += 1
                else:
                    nouns_in_sent += noun

            insert_value = (review_id, nouns_in_sent)
            insert_query = ("INSERT INTO nouns_list_per_sentence "
                            "(review_id, nouns)"
                            "VALUES (%s, %s)")
            cursor.execute(insert_query, insert_value)
    connection.commit()


def fetch_nouns_per_sentence():
    select_sql = 'SELECT * from nouns_list_per_sentence'
    cursor.execute(select_sql)
    return cursor.fetchall()


def insert_single_candidate_aspect_per_row(candidate_aspects):
    trucate_table('candidate_aspects')
    for review_id, can_asp in candidate_aspects:
        if can_asp:
            for cand_asp in can_asp:
                insert_value = (review_id, cand_asp)
                insert_query = ("INSERT INTO candidate_aspects "
                                "(review_id, candidate_aspect)"
                                "VALUES (%s, %s)")
                cursor.execute(insert_query, insert_value)
    connection.commit()

def insert_frequent_1_itemsets(frequent_1_itemset):
    trucate_table('frequent_itemsets')
    for key, value in frequent_1_itemset.items():
        insert_query = ("INSERT INTO frequent_itemsets "
                        "(frequent_itemsets)"
                        "VALUES (%s)")
        cursor.execute(insert_query, key)
    connection.commit()


def insert_frequent_k_itemsets(frequent_itemset):
    trucate_table('frequent_itemsets_k')
    for L in frequent_itemset:
        for key, value in frequent_itemset[L].items():
            fq_item = str(key).strip('')
            freq_item = eval(fq_item)
            insert_value = ' '.join(freq_item)
            insert_query = ("INSERT INTO frequent_itemsets_k "
                            "(frequent_itemsets)"
                            "VALUES (%s)")
            cursor.execute(insert_query, insert_value)
    connection.commit()


def fetch_frequent_itemsets():
    select_sql = 'SELECT frequent_itemsets FROM tcc.frequent_itemsets;'
    cursor.execute(select_sql)
    return [x[0] for x in cursor.fetchall()]


def fetch_frequent_k_itemsets():
    select_sql = 'SELECT frequent_itemsets FROM tcc.frequent_itemsets_k;'
    cursor.execute(select_sql)
    return [x[0] for x in cursor.fetchall()]


def insert_final_candidate_aspects(frequent_itemset):
    trucate_table('candidate_aspects_final')
    for cand_item in frequent_itemset:
        insert_query = ("INSERT INTO candidate_aspects_final "
                        "(aspect)"
                        "VALUES (%s)")
        cursor.execute(insert_query, cand_item)
    connection.commit()


def fetch_final_candidate_aspects():
    select_sql = 'SELECT aspect FROM tcc.candidate_aspects_final;'
    cursor.execute(select_sql)
    return [x[0] for x in cursor.fetchall()]


def insert_features_after_remove_aspects(features_set):
    trucate_table('pruning')
    for feature in features_set:
        insert_query = ("INSERT INTO pruning "
                        "(candidate_features)"
                        "VALUES (%s)")
        cursor.execute(insert_query, feature)
    connection.commit()


def fetch_features_after_remove_aspects():
    select_sql = 'SELECT candidate_features FROM tcc.pruning;'
    cursor.execute(select_sql)
    return [x[0] for x in cursor.fetchall()]

def trucate_table(table_name):
    truncate_table_sql = 'TRUNCATE TABLE ' + table_name
    cursor.execute(truncate_table_sql)
    connection.commit()

def fetch_sentence_by_id(sent_id):
    sql_query = "SELECT sentence FROM tcc.sentences WHERE review_id = '" + sent_id + "';"
    cursor.execute(sql_query)
    return [x[0] for x in cursor.fetchall()]


def insert_final_product_aspect_list(product_aspects_list_final):
    trucate_table('product_aspects')
    for aspect in product_aspects_list_final:
        insert_query = ("INSERT INTO product_aspects "
                        "(aspect)"
                        "VALUES (%s)")
        cursor.execute(insert_query, aspect)
    connection.commit()


def fetch_final_product_aspect_list():
    select_sql = 'SELECT aspect FROM tcc.product_aspects;'
    cursor.execute(select_sql)
    return [x[0] for x in cursor.fetchall()]


def insert_sentiment_analysis_result(sentiment_analysis_result_insert_into_db):
    trucate_table('sentiment_analysis')
    for aspect in sentiment_analysis_result_insert_into_db:
        insert_value = (aspect[0], aspect[1], aspect[2], aspect[3], str(aspect[4]).strip('[]'), str(aspect[5]).strip('[]'), str(aspect[6]).strip('[]'))
        insert_query = ("INSERT INTO sentiment_analysis "
                        "(product_aspect, pos_score, neg_score, neu_score, pos_sent_ids, neg_sent_ids, neu_sent_ids)"
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)")
        cursor.execute(insert_query, insert_value)
    connection.commit()

def fetch_sentiment_analysis():
    select_sql = 'SELECT * FROM tcc.sentiment_analysis;'
    cursor.execute(select_sql)
    return [x for x in cursor.fetchall()]