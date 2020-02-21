import codecs
import json
import os
import logging
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from . import utils
from collections import defaultdict

"""
Base and father class of graph-based extraction methods
"""

class textBase(object):
    def __init__(self, str_input=None, input_file=None):
        """
        build raw text to be extracted from string-format or txt file 
        """
        self.raw_text = ""
        if str_input:
            self.raw_text = str_input
        if input_file:
            # Read from txt file
            if os.path.isfile(input_file):
                self.raw_text = utils.read_from_txt(input_file)
            else:
                logging.exception("Invalid file path")
        # weights used to rank the phrase or words
        self.weights = defaultdict(float)
    
    """
    Member functions:
    Generate candidates nodes in graph
    Generate candidates phrases
    Rank phrases
    """
    def get_word_list(self):
        """
        Build a list of words that *KEEPS* stop words and punctuations
        (Used for window distance checking)
            Return
                List of words
        """
        return utils.generate_window_words(self.raw_text)

    def get_node_list(self):
        """
        Noun or adjective as graph nodes
            Return
                List of nodes
        """
        return utils.generate_graph_nodes(self.raw_text)
    
    def get_candidate_phrases(self):
        """
        Example
            ["round neck", "High waist"]
        """
        return utils.generate_candidate_phrase(self.raw_text)
    
    def get_top_phrases(self, k=6):
        """
        Rank self.weights variable and return top k in form of list of phrases
        """
        sorted(self.weights.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        top_phrase = [ name for name, value in self.weights.items()]
        top_amount = min(len(top_phrase), k)
        return top_phrase[:top_amount]

'''
test_str = "He's name is John, he've lived in London for long time."
base = textBase(test_str)
wl = base.get_word_list()
nl = base.get_node_list()
print(wl)
print(nl)
'''
