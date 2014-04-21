import numpy
import pickle
class LDA:
    # Initialize with set up value
    def initialize(self, counts, k, n_d_k, n_k_w, n_k, topic_per_word):
        self.k = k
        self.count = counts
        self.n_k_w = n_k_w
        self.n_d_k = n_d_k
        self.n_k = n_k
        self.topic_per_word = topic_per_word

    def __init__(self, k, file_name):
        self.k = k
        self.counts = 0
        # self.k is the number of topics
        f = open(file_name, "r")
        lines = f.readlines()  # Reads lines of the file and stores it in a list
        lines = filter(None, [line.strip(' \n\t') for line in lines])  # Strips the lines
        self.documents = dict()
        self.vocabulary = set()

        # Reading all self.documents
        for line in lines:
            l = filter(None,line.strip('\n\t ').split(' '))
            doc_id = l[0]
            self.documents[doc_id] = l[1:] # Words in the document
            self.vocabulary = self.vocabulary.union(l[1:]) # Update self.vocabulary
        #We have the self.documents in memory now!
        self.topic_per_word = dict() #For each document, you have a list of topics here

        #Initializing all counts
        self.n_d_k = dict() # Per document, how many words a topic: For a self.key, get a list
        self.n_k_w = [] # Per topic, how many instances of a word: For an idnex, get a dictionary.
        # For a word self.key get an integer
        self.n_k = [0]*self.k # No of words a topic
        self.beta = dict() # Per word, how many prior instances of a word: get a dictionary.
        self.beta_sum = 0
        self.alpha = [2]*self.k # ALPHA value

        for word in self.vocabulary:
            self.beta[word] = 2
            self.beta_sum += 2

        for topic in xrange(self.k):
            dummy = dict()
            for word in self.vocabulary:
                dummy[word] = 0
            self.n_k_w += [dummy]

        for doc_id in self.documents.keys():
            words_in_doc = self.documents[doc_id]
            #Get all the words
            #Generate topic assignment
            topic_for_doc = []
            words_per_topic = [0]*self.k
            for word in words_in_doc:
                topic = numpy.random.randint(0, self.k) #Random initialization of topic
                topic_for_doc += [topic]
                self.n_k[topic] += 1
                self.n_k_w[topic][word] += 1
                words_per_topic[topic] += 1
            self.n_d_k[doc_id] = words_per_topic
            self.topic_per_word[doc_id] = topic_for_doc

    def run(self, iterations, output_file):
        for x in xrange(iterations):
            for doc_id in self.documents.keys():
                for i in xrange(len(self.documents[doc_id])):
                    word = self.documents[doc_id][i]
                    topic = self.topic_per_word[doc_id][i]
                    self.n_d_k[doc_id][topic] -= 1
                    self.n_k_w[topic][word] -= 1
                    self.n_k[topic] -= 1
                    #We'v unset the topic assigned
                    probab = []
                    for t in xrange(self.k):
                        p = self.n_d_k[doc_id][t] + self.alpha[t]
                        q = self.n_k_w[t][word] + self.beta[word]
                        r = self.n_k[t] + self.beta_sum
                        probab += [float(p*q)/r]
                    pr = [p/sum(probab) for p in probab]
                    topic = list(numpy.random.multinomial(1, pr)).index(1)
                    self.topic_per_word[doc_id][i] = topic #Reassign new topic
                    # Update counts accordingly
                    self.n_d_k[doc_id][topic] += 1
                    self.n_k_w[topic][word] += 1
                    self.n_k[topic] += 1
            print x
        self.counts += iterations
        #Write class to file
        f = open(output_file, "w")
        pickle.dump(self, f)
        f.close()




