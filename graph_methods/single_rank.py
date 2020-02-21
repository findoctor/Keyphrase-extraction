#from __future__ import absolute_import, division, print_function
from .. import base
import networkx as nx


class singleRank(base.textBase):
    def __init__(self, str_input=None, input_file=None):
        super(singleRank, self).__init__(str_input, input_file)
        self.graph = nx.Graph()

    def build_graph(self, window=3):
        word_list = self.get_word_list()
        node_list = self.get_node_list()
        # Add nodes
        self.graph.add_nodes_from(node_list)
        # Add edges
        for i in range(len(word_list)):
            w1 = word_list[i]
            for j in range(i+1, min(i+window, len(word_list)) ):
                w2 = word_list[j]
                if not self.graph.has_edge(w1, w2):
                    self.graph.add_edge(w1,w2,weight=0.0)
                else:
                    self.graph[w1][w2]['weight'] += 1.0

    def weight_node(self):
        """
        Run PageRank with weighted edges
        Build self.weights
        """
        self.build_graph()
        # pr is dictionary of nodes with PageRank as value
        pr = nx.pagerank(self.graph, alpha=0.85, tol=1e-7, weight='weight')
        self.candidate_phrases = self.get_candidate_phrases()

        for phrase in self.candidate_phrases:
            consist_words = phrase.split()
            # Phrase score
            # valid_length: # words of phrase included in graph
            p_score = 0.0
            valid_length = 0
            for word in consist_words:
                if word in pr:
                    valid_length+=1
                    p_score+=pr[word]
            p_score /= valid_length
            self.weights[phrase] = p_score 


'''
test_str = "He's name is John, he've lived in London for long time. John is teacher and happy."
singlerank = singleRank(str_input=test_str)
singlerank.weight_node()
res = singlerank.get_top_phrases()
print(res)
'''