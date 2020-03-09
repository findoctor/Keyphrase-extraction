from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import json
from .config import DST_PATH
import codecs
import random

def read_from_txt(path):
    # return string as raw text
    with open(path, 'r') as file:
        text = file.read().replace('\n', '.\n')
    return text

def cosine_sim(emb1, emb2):
    return 1 - spatial.distance.cosine(emb1, emb2)

def _MMR(k, word_embs, doc_emb, beta=0.5):
    '''
    ## Input
    - word_embs: list of phrase-embedding pair
    - k: number of keyphrases we want to extract
    - beta: params for MMR
    
    ## Output: list of keyphrases
    '''
    selected = []
    unselected = word_embs[:]
    
    k = min(k, len(unselected))
        
    while(len(selected) < k):
        max_sim = -1.0
        if len(unselected) <= 1:
            break
        #added = unselected[0]
        added = None
        for tup in unselected:
            phrase = tup[0]
            emb = tup[1]
            max2selectedSim = 0
            for s_tup in selected:
                s_phra, s_emb = s_tup[0], s_tup[1]
                Sim2selected = cosine_sim(s_emb, emb)
                if Sim2selected > max2selectedSim:
                    max2selectedSim = Sim2selected
            cur_sim = beta*cosine_sim(emb, doc_emb)-(1-beta)*(max2selectedSim)
            if cur_sim > max_sim:
                max_sim = cur_sim
                added = (phrase, emb)
        # print("We add "+ str(added[0]) + " to the selected" )
        if added:
            selected.append(added)
            
        if added in unselected:
            unselected.remove(added)
        #print(added[0])
        #print("k="+ str(len(selected)))
    return [_str for _str, emb in selected]

# Randomly choose a piece of description from a vendor
def select_from_vendor(vendor):
    path = DST_PATH
    handle = codecs.open(path, 'r', encoding='utf8')
    json_decode=json.load(handle)
    candidates = []  # text pool
    for item in json_decode:
        if(str(item.get('vendor') ) == vendor ):
            candidates.append(item.get('description'))
    # Randomly choose an element from list
    text = random.choice(candidates)
    print("* * * * * * * * * * * * * * selected description * * * * * * * * * * * * * * ")
    print(text)
    return text