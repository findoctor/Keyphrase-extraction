import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import codecs
from . import config as cfg

def read_from_txt(path):
    # return string as raw text
    with open(path, 'r') as file:
        text = file.read().replace('\n', '.\n')
    return text

def extract_phrases(my_tree, phrase):
    """
    Helper function for generating candidate phrases
    """
    my_phrases = []
    if my_tree.label() == phrase:
        my_phrases.append(my_tree.copy(True))

    for child in my_tree:
        if type(child) is nltk.Tree:
            list_of_phrases = extract_phrases(child, phrase)
            if len(list_of_phrases) > 0:
                my_phrases.extend(list_of_phrases)

    return my_phrases

def generate_candidate_phrase(test_sentence, grammers = cfg.GRAMMER):
    words = nltk.word_tokenize(test_sentence)
    tags = nltk.pos_tag(words)
    candidate_phrases = set([])
    for grammar in grammers:
        parser = nltk.RegexpParser(grammar)
        tree = parser.parse(tags)
        candidate_trees = extract_phrases(tree, 'NP')
        for phrase in candidate_trees:
            candidate_phrases.add(" ".join([x[0].lower() for x in phrase.leaves()]))   # Lower case
    return list(candidate_phrases)

# adj or noun as graph nodes (usd for TPR, pageRank)
def generate_graph_nodes(test_sentence, grammar = cfg.adj_or_noun):
    words = nltk.word_tokenize(test_sentence)
    tags = nltk.pos_tag(words)
    candidate_words = set([])

    parser = nltk.RegexpParser(grammar)
    tree = parser.parse(tags)
    candidate_trees = extract_phrases(tree, 'NP')
    for phrase in candidate_trees:
        candidate_words.add("".join([x[0].lower() for x in phrase.leaves()]))
    return list(candidate_words)

def clean_text(doc):
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    result = tokenizer.tokenize(doc)
    new_sentence =[]
    for w in result:
        if w not in stop_words:
            new_sentence.append(w)
    new_sentence = " ".join(item for item in new_sentence)
    return new_sentence

def generate_window_words(doc):
    # Avoiding "He's" be tokenized as He + 's
    # tokenizer = nltk.tokenize.MWETokenizer()
    result = nltk.word_tokenize(doc)
    new_sentence =[]
    for w in result:
        new_sentence.append(w)
    return new_sentence
