import json
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from DataProcess import clean_text

# TODO: Preprocess data: remove stop words, ...

class LDA(object):
    def __init__(self, path, no_topics=6, no_features=1000):
        """
        no_features is subject to the size of vocab
        """
        self.corpus_path = path
        self.no_features = no_features
        self.no_topics = no_topics
        print("Start building corpus")
        self.convert_data()
    
    def convert_data(self):
        corpus = []
        raw_text =[]
        handle=codecs.open(self.corpus_path, 'r', encoding='utf8')
        json_decode=json.load(handle)

        num = 1
        for item in json_decode:
            if item.get('description'):
                # Clean text, remain raw data to build POS tagging
                doc = clean_text(item.get('description'))
                corpus.append(doc)
                raw_text.append( item.get('description') )
                num+=1
        print("Total number of docs we use: " + str(num) )
        self.corpus = corpus
        self.raw_text = raw_text

    ''' Return:
    1. the topic-vocabulary distribution
    2. the document-topics distribution
     '''
    def lda_func(self):
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        # Revision: set max_features from 1000 to None
        tf = tf_vectorizer.fit_transform(self.corpus)
        tf_feature_names = tf_vectorizer.get_feature_names()
        lda = LatentDirichletAllocation(n_topics=self.no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
        lda_model = lda.fit(tf)
        lda_doc2topic = lda.transform(tf)
        self.model = lda_model
        return lda_model.components_, lda_doc2topic, tf_feature_names 
        # 1.(#Topics, #Vocabulary) P_tw
        # 2.(#Document, #Topics)   Pt


    def display_topics(self, feature_names, no_top_words=10):
        for topic_idx, topic in enumerate(self.model.components_):
            print("Topic %d:" % (topic_idx) )
            print("  ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]) )

    
