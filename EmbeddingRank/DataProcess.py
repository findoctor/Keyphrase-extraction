# Extract english contents from the json file
# Vendors: xxx_uk (contains _uk)
# We exclude Tigerofsweden_uk because the description is already well-structured key phrases format
# Store the description in DST_PATH as a list of dictionaries
import json
import nltk
import spacy
import codecs
from . import config as cfg

# nltk.download('averaged_perceptron_tagger')

def extract_phrases(my_tree, phrase):
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
    # Replace \n with .\n
    test_sentence = test_sentence.replace('\n', '.\n')
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
        candidate_words.add("".join([x[0] for x in phrase.leaves()]))
    return list(candidate_words)

#if __name__ == '__main__':
#   pass
    # Build dataset
    # build_dataset(cfg.SRC_PATH, cfg.DST_PATH)