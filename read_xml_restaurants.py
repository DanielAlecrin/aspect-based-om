from bs4 import BeautifulSoup
import pandas as pd

def read_semeval2016_task5_subtask1(filepath):
    reviews = []
    reviewCount = 0
    with open(filepath) as f:
        soup = BeautifulSoup(f, "xml")
        review_tags = soup.find_all("Review")

        final_list = []
        for j, r_tag in enumerate(review_tags):

            reviewId = r_tag["rid"]
            sentence_tags = r_tag.find_all("sentence")
            for s_tag in sentence_tags:

                sentenceId = s_tag["id"]
                sentenceText = s_tag.find("text").get_text()
                try:
                    if s_tag["OutOfScope"]:
                        continue
                except KeyError:
                    pass

                opinion_tags = s_tag.find_all("Opinion")
                for o_tag in opinion_tags:
                    # polarity
                    opinionPolarity = ''
                    try:
                        opinionPolarity = o_tag["polarity"]
                    except KeyError:
                        opinionPolarity = None

                    pol = 0
                    if opinionPolarity == 'positive':
                        pol = 1
                    elif opinionPolarity == 'negative':
                        pol = -1
                    elif opinionPolarity == None:
                        continue

                    # target
                    opinionTarget = ''
                    try:
                        opinionTarget = o_tag["target"]
                        if opinionTarget == "NULL":
                            continue
                    except KeyError:
                        pass

                    reviewCount = reviewCount + 1
                    
                    final_list.append(("_" + str(reviewCount), sentenceId, sentenceText, opinionTarget, pol))
    return final_list

# Train
reviews = read_semeval2016_task5_subtask1("data/ABSA16_Restaurants_Train_SB1_v2.xml")
df = pd.DataFrame(reviews, columns=['reviewId', 'sentenceId', 'text', 'aspect_term', 'polarity'])
df.to_csv('data/data_train.csv', index=None)

# Test
reviews = read_semeval2016_task5_subtask1("data/EN_REST_SB1_TEST.xml.gold")
df = pd.DataFrame(reviews, columns=['reviewId', 'sentenceId', 'text', 'aspect_term', 'polarity'])
df.to_csv('data/data_test.csv', index=None)
