from .. import base
import networkx as nx
from .single_rank import singleRank
import os
import pickle
import numpy as np
from scipy.spatial.distance import cosine

class singleTPR(singleRank):
    def __init__(self, doc_index, str_input=None, input_file=None):
        super(singleTPR, self).__init__(str_input, input_file)
        self.graph = nx.Graph()
        self.doc_index = doc_index

    def load_LDA(self, LDA_path="pke/LDA"):
        """
        Return what we get from LDA: (Loaded from pkl, trained on Norna dataset)
            1. Topic to word distribution (t2w)
            2. Document to topic distribution (d2t)
            3. Feature names (fn)
        """
        t2w_path = os.path.join(LDA_path, "topic2voc_clean.pickle")
        d2t_path = os.path.join(LDA_path, "doc2topic_clean.pickle")
        fn_path = os.path.join(LDA_path, "feature_names_clean.pickle")

        handle1 = open(t2w_path,"rb")
        self.t2w = pickle.load(handle1)

        handle2 = open(d2t_path,"rb")
        self.d2t = pickle.load(handle2)

        handle3 = open(fn_path,"rb")
        self.fn = pickle.load(handle3)

    def weight_node(self):
        self.load_LDA()
        self.candidate_phrases = self.get_candidate_phrases()
        self.build_graph(window=2)

        # Check dimension
        print("Dimension of t2w:")
        print(self.t2w.shape )
        print("Dimension of d2t:")
        print(self.d2t.shape )

        # Compute w_i :topical importance of each word
        K = len(self.t2w)
        W = {}
        for word in self.graph.nodes():
            if word in self.fn:
                index = self.fn.index(word)
                distribution_word_topic = [self.t2w[k][index] for k in range(K)]
                distribution_doc_topic = self.d2t[self.doc_index]
                W[word] = 1 - cosine(distribution_word_topic, distribution_doc_topic)

        # get the default probability for OOV words
        default_similarity = min(W.values())
        for word in self.graph.nodes():
            if word not in W:
                W[word] = 0.0

        # Normalize the topical word importance of words
        norm = sum(W.values())
        for word in W:
            W[word] /= norm

        # compute the word scores using biased random walk
        pr= nx.pagerank(G=self.graph,
                        personalization=W,
                        alpha=0.85,
                        tol=0.0001,
                        weight='weight')
        
        # Rank phrases
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



test_str = "Chunky beanie crafted in a super soft cashmere and wool quality. It has a double folded edge and is branded with an embroidered logo at front. \
    Ribbed wool and cashmere quality. Double folded edge. 57cm- Branded with JLJL logo. Unisex style. No. FMAC01250."
tpr_rank = singleTPR(doc_index = 0, str_input=test_str)
tpr_rank.weight_node()
res = tpr_rank.get_top_phrases()
print(res)

