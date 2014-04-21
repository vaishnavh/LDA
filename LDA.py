import numpy
# Gibbs sampling!! Yay
for x in xrange(10):
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
            pr = [p/sum(probab) for p in probab]
            topic = list(numpy.random.multinomial(1, pr)).index(1)
            topic_per_word[doc_id][i] = topic #Reassign new topic
            # Update counts accordingly
            n_d_k[doc_id][topic] += 1
            n_k_w[topic][word] += 1
            n_k[topic] += 1
    print x











