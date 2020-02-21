import sent2vec
import nltk
import json
#import config as cfg
import codecs
from .utils import select_from_vendor, cosine_sim
from .DataProcess import generate_candidate_phrase
from .utils import _MMR
from . import utils
from .embeddings import sentence_embedding
import string
import logging
import os

class EmbedRank(object):
    def __init__(self, k=6, embedModel='wiki_unigrams.bin'):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(embedModel)
        self.k = k

    def load_text(self, doc=None, input_file=None):
        if doc and input_file:
            logging.exception("Don't input raw text and file path at the same time")
        elif doc:
            self.raw_text = doc
        elif input_file:
            if os.path.isfile(input_file):
                self.raw_text = utils.read_from_txt(input_file)
            else:
                logging.exception("Invalid file path")
        tokens = nltk.word_tokenize(self.raw_text)
        doc_str = " ".join(item for item in tokens if item not in string.punctuation)
        self.sentence = doc_str   # used for embedding
        self.candidate_phrases = generate_candidate_phrase(self.raw_text)

    @classmethod
    def from_vendor(cls, vendor):
        test_sentence = select_from_vendor(vendor)
        cls.doc = test_sentence
        candidate_phrases = generate_candidate_phrase(test_sentence)
        cls.candidate_phrases = candidate_phrases

        '''
        # Ensemble all phrases in one sentence
        doc_str = ""
        for cand in candidate_phrases:
            doc_str += cand
            doc_str += ' '
        '''
        sentences = nltk.word_tokenize(test_sentence)
        doc_str = " ".join(item for item in sentences if item not in string.punctuation)
        
        cls.sentence = doc_str
        wrapper = cls(doc_str)
        return wrapper

    def topK_phrase(self):
        # doc_emb = sentence_embedding(self.doc)
        doc_emb = self.model.embed_sentence(self.sentence)  # sentence embedding
        # word-embedding of all candidate phrases
        word_embs = []  # contain phrase-embedding pair

        for candidate_phrase in self.candidate_phrases:
            if ' ' in candidate_phrase:  # if phrase
                cand_emb = self.model.embed_sentence(candidate_phrase)
            else:
                cand_emb = self.model.embed_unigrams([candidate_phrase])
            word_embs.append((candidate_phrase, cand_emb))
        return _MMR(self.k, word_embs, doc_emb)


'''
# Test Code
test_str = "Chunky beanie crafted in a super soft cashmere and wool quality. It has a double folded edge and is branded with an embroidered logo at front. \
    Ribbed wool and cashmere quality. Double folded edge. 57cm- Branded with JLJL logo. Unisex style. No. FMAC01250."
path_to_sen2vecmodel = 'wiki_unigrams.bin'
emb_rank = EmbedRank(k=5, embedModel = path_to_sen2vecmodel)
emb_rank.load_text(test_str)
res = emb_rank.topK_phrase()
print(res)
'''

    

