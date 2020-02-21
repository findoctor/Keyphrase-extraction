# Keyphrase extraction

Keyphrase extraction meant to extract meaningful attributes from fashion products. Python3 based.
What implemented so far:
* Graph based methods
* Embedding based methods

TODO:
* Remove meaningless but frequent words and rebuild the LDA model
* Maybe try some DL supervised methods

## Installation of dependencies
```bash
pip3 install -r requirements.txt
```
```bash
python -m nltk.downloader stopwords
python -m nltk.download('averaged_perceptron_tagger')
```

To enable the use of Embedding-based method, follow innstructions [here](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models)
to download Sent2vec model and store it in the 'EmbeddingRank' folder. Name it as 's2v.bin' for example. <br>

Since I have already trained with LDA, I stored the word-to-topic distribution, document-to-topic distribution and feauture_names
as pickle file inside LDA folder. Feel free to train with your own corpus using lda.py and replace them with your trained distributions.


## Minimal example
After download the whole repo and named it as kkExtract:
```python
# Example of single-tpr extraction
import kkExtract
# tpr_rank = kkExtract.graph_methods.singleTPR(doc_index = 12, str_input=test_str) # you need to specify the document index in the corpus, build from string
tpr_rank = singleTPR(doc_index = 12, input_file=path/to/your/txt_file) # build from txt file
tpr_rank.weight_node()
res = tpr_rank.get_top_phrases(k=6)
print(res)
```

```python
# Example of EmbedRank extraction
path_to_sen2vecmodel = 's2v.bin'
emb_rank = EmbedRank(k=5, embedModel = path_to_sen2vecmodel)
emb_rank.load_text(test_str)
res = emb_rank.topK_phrase()
```

Detailed examples are provided in the end of each methods python file


Currently implements the following keyphrase extraction models:

* Graph-based models
  * SingleRank  [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#singlerank), [article by (Wan and Xiao, 2008)](http://www.aclweb.org/anthology/C08-1122.pdf)]
  * Single Topical PageRank [[documentation](https://dl.acm.org/doi/abs/10.1145/2740908.2742730)]
* Emdedding-based models
  * EmbedRank [[documentation](https://www.aclweb.org/anthology/K18-1022) ]



