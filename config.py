BATCH_SIZE = 128
K = 6   # number of keyphrase we intend to extract

# GRAMMER =  "NP:{<JJ.*><NN.*>+}"
# adj_noun =  "NP:{<JJ.*>*<JJ.*>*<NN.*>*}"
adj_noun =  "NP:{<CD>*<JJ.*>*<JJ.*>*<NN.*>*}"
adj_conj_noun = "NP:{<JJ.*>+<CC><JJ.*>*<NN.*>*}"
adv_verb = "NP:{<RB.*><VB.*>}"
adj_to_verb = "NP:{<JJ.*>*<TO><VB.*>}"

adj_or_noun =  "NP:{<JJ.*> | <NN.*>}"

GRAMMER = [adj_noun, adj_conj_noun, adv_verb, adj_to_verb]