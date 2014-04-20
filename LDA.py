import numpy
'''
topic_per_word = None
n_d_k = None
n_k_w = None
n_k = None
vocabulary = None
'''

k = 5
file_name = "docs.txt"

# k is the number of topics
f = open(file_name, "r")
lines = f.readlines()  # Reads lines of the file and stores it in a list
lines = filter(None, [line.strip(' \n\t') for line in lines])  # Strips the lines
documents = dict()
vocabulary = set()

# Reading all documents
for line in lines:
    l = filter(None,line.strip('\n\t ').split(' '))
    doc_id = l[0]
    documents[doc_id] = l[1:] # Words in the document
    vocabulary = vocabulary.union(l[1:]) # Update vocabulary
#We have the documents in memory now!
topic_per_word = dict()

#Initializing all counts
n_d_k = dict() # Per document, how many words a topic: For a key, get a list
n_k_w = [] # Per topic, how many instances of a word: For an idnex, get a dictionary.
# For a word key get an integer
n_k = [0]*k # No of words a topic
beta = dict() # Per word, how many prior instances of a word: get a dictionary.
beta_sum = 0
alpha = [2]*k # ALPHA value

for word in vocabulary:
    beta[word] = 2
    beta_sum += 0

for topic in xrange(k):
    dummy = dict()
    for word in vocabulary:
        dummy[word] = 0
    n_k_w += [dummy]

for doc_id in documents.keys():
    words_in_doc = documents[doc_id]
    #Get all the words
    #Generate topic assignment
    topic_for_doc = []
    words_per_topic = [0]*k
    for word in words_in_doc:
        topic = numpy.random.randint(0, k) #Random initialization of topic
        topic_for_doc += [topic]
        n_k[topic] += 1
        n_k_w[topic][word] += 1
        words_per_topic[topic] += 1
        topic_for_doc += [topic]
    n_d_k[doc_id] = words_per_topic
    topic_per_word[doc_id] = topic_for_doc

# Gibbs sampling!! Yay
for x in xrange(50):
    for doc_id in documents.keys():
        for i in xrange(len(documents[doc_id])):
            word = documents[doc_id][i]
            topic = topic_per_word[doc_id][i]
            n_d_k[doc_id][topic] -= 1
            n_k_w[topic][word] -= 1
            n_k[topic] -= 1
            #We'v unset the topic assigned
            probab = []
            for t in xrange(k):
                p = n_d_k[doc_id][t] + alpha[t]
                q = n_k_w[t][word] + beta[word]
                r = n_k[t] + beta_sum
                probab += [float(p*q)/r]
            probab = probab/sum(probab)
            topic = list(numpy.random.multinomial(1, probab)).index(1)
            n_d_k[doc_id][topic] += 1
            n_k_w[topic][word] += 1
            n_k[topic] += 1











