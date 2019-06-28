from stanfordcorenlp import StanfordCoreNLP
import json


class SNLPServer:
    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=2000)

        self.props = {
            'annotators': 'pos',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def close(self): 
        self.nlp.close()
