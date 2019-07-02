import database
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from itertools import combinations

def main():
    features_list_database = database.fetch_final_product_aspect_list()
    features_list = []
    for f in features_list_database:
        if len(f.split()) == 1 and f not in features_list:
            features_list.append(f)
        else:
            redundant_aspect_list = []
            for i in range(len(f.split())):
                combine = combinations(f.split(), i + 1)
                for c in combine:
                    redundant_aspect_list.append(" ".join(c))

            for asp in redundant_aspect_list:
                if asp not in features_list:
                    features_list.append(asp)

    sentence_in_reviews = database.fetach_pos_tagged_sentence()
    sentence_containing_feature_word = []
    for review_id, sentences in sentence_in_reviews:
        sentence = eval(sentences)
        prev_word = ''
        #print(sentence)
        for word, tag in sentence:

            #print('word: ', word)
            current_word = ''
            for feature in features_list:
                if len(feature.split()) == 1:
                    if feature == word:
                        sentence_containing_feature_word.append((feature, review_id))
                else:
                    complete_word = ''
                    #print('teste', feature)
                    for i in range(len(feature.split())):
                        if feature.split()[i] == word:
                            if prev_word != '':
                                current_word = prev_word + ' ' + feature.split()[i]
                            else:
                                current_word = feature.split()[i]
                            prev_word = current_word
                    #print('depois', current_word)
                    if len(current_word) != 1 and current_word != '':
                        complete_word = current_word
                    #print('complete word: ', complete_word)
                    if complete_word == feature:
                        sentence_containing_feature_word.append((feature, review_id))

    
    #print(sentence_containing_feature_word)

    feature_list_ids = {}
    for feature_word, review_id in sentence_containing_feature_word:
        if feature_word not in feature_list_ids:
            feature_list_ids[feature_word] = {}
            feature_list_ids[feature_word]['Review_ID'] = []
            feature_list_ids[feature_word]['Review_ID'].append(review_id)
            #feature_list_ids.append((feature_word))
        else:
            feature_list_ids[feature_word]['Review_ID'].append(review_id)

    with open('data/data_train.csv', 'r', encoding='utf8') as csvfile:
        review_list = pd.read_csv(csvfile)
    
    classifier = pickle.load(open('models/nb_classifier.sav', 'rb'))
    vectorizer = pickle.load(open('models/nb_vectorizer.sav', 'rb'))

    text_vector = vectorizer.transform(review_list['text'])
    predicted_texts = classifier.predict(text_vector)

    review_ids_dataset = []
    count = 0
    for item in review_list['reviewId']:
        review_ids_dataset.append((item, predicted_texts[count]))
        count = count + 1

    review_polarity_predicted = pd.DataFrame(review_ids_dataset, columns=['reviewId', 'polarity'])

    final_list = []
    for feature_key, feature_values in feature_list_ids.items():
        #print(feature_key)
        #print(len(feature_values['Review_ID']))
        pos_value = 0
        neg_value = 0
        for review_id in feature_values['Review_ID']:
            registry = review_polarity_predicted[review_polarity_predicted['reviewId'] == review_id]

            if len(registry) > 0:
                registry = registry.iloc[0]['polarity']
                #print(registry)
                if registry == 1:
                    pos_value = pos_value + 1
                else:
                    neg_value = neg_value + 1
        final_list.append((feature_key, pos_value, neg_value))


    index_names = []
    pos_per = []
    neg_per = []
    for aspect, pos_value, neg_value in final_list:
        total = pos_value + neg_value

        #if neg_value > 3:
        pos_per.append((pos_value/total))
        neg_per.append((neg_value/total))
        index_names.append(aspect)

        #if len(index_names) == 10:
        #    break

    speed = [0.1, 17.5, 40, 48, 52, 69, 88]
    lifespan = [2, 8, 70, 1.5, 25, 12, 28]
    
    df = pd.DataFrame({'score positivo': pos_per,
                       'score negativo': neg_per 
                       }, index=index_names)
    ax = df.plot.barh(stacked=True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    plt.show()


if __name__ == '__main__':
    main()