import numpy as np
import sent2vec
import nltk
import re
import string

def sentence_embedding(doc, sent_model=sent2vec.Sent2vecModel()):
    """
    params: sent_model: sen2vec model
    """
    sentences = nltk.word_tokenize(doc)
    #re.sub("'t", 'ot', "n't, doesn't, can't, don't")
    res_sent = " ".join(item.lower() for item in sentences if item not in string.punctuation )
    #print(res_sent)
    #print(sent_model.embed_sentence(res_sent))
    #return sent_model.embed_sentence(res_sent)


def phrase_embedding(phrase):
    pass


# TEST:

#print(sentence_embedding("This is he's for test only, don't take \n it seriously!") )