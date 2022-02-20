import numpy as np
from numpy import dot
from numpy.linalg import norm

from collections import Counter

#define an embeddings file -- we use the GloVe.6B.50D here; which has 50-dim word embeddings for 400k words
embedding_file = '../glove.6B.50d.txt'

class GloVeEmbeddings:

    def __init__(self, embedding_file=embedding_file):
        self.embedding_file = embedding_file
        self.embeddings = {}

    def initialize_embeddings(self):

        with open(self.embedding_file) as embedding_file:

            for line in embedding_file:
                values = line.split()
                word = values[0]
                word_embedding = np.asarray([float(value) for value in values[1:]]) #by default it will be an array of string values instead of array of float values

                self.embeddings[word] = word_embedding

            self.embedding_size = len(values[1:])

        print('Initialized word embeddings from {} file'.format(self.embedding_file))
        print('Total words : {}'.format(len(self.embeddings)))
        print('Embedding size : {}'.format(self.embedding_size))

class WordEmbeddingTopicCoherence(GloVeEmbeddings):
    """
    Implementation of WETC from Amazon NTM blog for topic modeling, paper : https://arxiv.org/pdf/1809.02687.pdf

    Args:
        GloVeEmbeddings : class to initialize GloVe Embedding word vectors.
                          Get the GloVe embedding files from https://www.kaggle.com/anindya2906/glove6b; we use the glove.6B.50d.txt as default

    Run WordEmbeddingTopicCoherence()get_output(topic_words = topic_words) to get wetc score per list of topic words
    """

    def __init__(self, embedding_file=embedding_file, weigh=False):
        
        self.embedding_file = embedding_file
        
        self.topic_word_weights = {}
        self.weigh = weigh

        self.embeddings = {}
        self.embedding_size = 0

        #get weights from the pretrained embedding file
        super().__init__(embedding_file=self.embedding_file)
        
        #initialize all embeddings from pretrained embedding file
        self.initialize_embeddings()

    def initialize_word_weights(self, topic_words):

        all_topic_words = ' '.join(topic_words)
        unigram_tokens = all_topic_words.split()

        #get frequency of each unigram separately from topic words
        word_frequencies = Counter(unigram_tokens)

        #compute topic word weights as (frequency of a unigram / total number of topic words)
        for word in word_frequencies:
            self.topic_word_weights[word] = word_frequencies[word]/len(topic_words)

        print('Initialized topic word weights')
        print('word weights : {}'.format(self.topic_word_weights))

    def form_word_pairs(self, words):

        pairs = [[[words[i], words[j]] for j in range(i, len(words)) if words[i] != words[j]] for i in range(len(words))]
        
        #flatten
        pairs = [word_pair for pair in pairs for word_pair in pair]
        
        return pairs

    def form_average_word_embedding(self, token):
        
        words = token.split()
        average_embeddings = []
        
        for word in words:

            try:
                embedding = self.embeddings[word]
            except:
                embedding = [1]*self.embedding_size #default vector is [0, 0, 0, ..., 0] or [1, 1, 1, ... , 1]; we choose [1, 1, 1, ..., 1]

            if self.weigh:

                #for each unigram, weighted embedding will be (word embedding)*(weight of the word)
                weight = self.topic_word_weights[word]
                embedding = np.asarray([float(weight)*value for value in embedding]).astype('float32')
                del weight
            
            average_embeddings.append(embedding)
            del embedding
        
        average_embeddings = np.asarray(average_embeddings).astype('float32')
        average_embeddings = np.mean(average_embeddings, axis=0)

        return average_embeddings.astype('float32')

    def form_embedding(self, word):

        #if embedding for the word is not found, use a 1 valued vector as the embedding for it, treating it as an unknown token
        if word not in self.embeddings:

            num_tokens = len(word.split()) #find if it is a multi-word token

            if num_tokens > 1:
                print('Multi-word token found, forming average vector')
                word_embedding = self.form_average_word_embedding(token=word)

            else:
                word_embedding = np.ones(self.embedding_size).astype('float32')
                print('Embedding for word {} not found -- treating word as a <UNK> token'.format(word))
                print('Embedding : {}'.format(word_embedding))

            self.embeddings[word] = word_embedding
        
        #return embedding of the word
        return self.embeddings[word]

    def compute_cosine_similarity(self, pair):

        a, b = np.array(pair[0]).astype('float32'), np.array(pair[1]).astype('float32')
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        
        return cos_sim

    def compute_pairwise_cosine_similarity(self, words):

        #form word pairs
        word_pairs = self.form_word_pairs(words)

        #form array of embeddings for each word pair
        embeddings_for_word_pairs = [[self.form_embedding(words[0]), self.form_embedding(words[1])] for words in word_pairs]

        #compute similarity for each word pair
        embedding_coherence = [self.compute_cosine_similarity(embedding_pair) for embedding_pair in embeddings_for_word_pairs]

        return embedding_coherence

    def get_output(self, topic_words):

        #initialize weights for all unigrams IF weighing is set to True
        if self.weigh:
            self.initialize_word_weights(topic_words=topic_words)

        #get cosine similarity for all word pairs
        embedding_coherence = self.compute_pairwise_cosine_similarity(topic_words)

        #average all of the cosine similarities
        word_embedding_topic_coherence = np.mean(np.asarray(embedding_coherence))

        #return this score for the topic words as the WETC score
        print('{} -> {}'.format(word_embedding_topic_coherence, topic_words))
        return word_embedding_topic_coherence
        
class TopicUniqueness:
    """
    Implementation of the Topic Uniqueness metric from the Amazon NTM blog : https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-neural-topic-model-now-supports-auxiliary-vocabulary-channel-new-topic-evaluation-metrics-and-training-subsampling/#:~:text=Word%20embedding%20topic%20coherence%20metric,top%20words%20in%20each%20topic.

    Topic Uniqueness is a measure of how 'unique' a topic is compared to all the extracted topics.
    Topics with low topic uniqueness have a chance to be merged to form a bigger topic

    TU implementation works best for unigram topic words.

    run TopicUniqueness(topics = topics).get_output() to get uniqueness for each topic. topics = topic words per topic, for all topics
    """

    def __init__(self, topics):

        self.topics = topics
        self.K = len(topics)
        self.n = len(self.topics[0])
        
        self.cnt_lookup = {}
        self.topic_uniqueness = {}
    
    def compute_cnt_per_topic(self, k):

        count = {word:0 for word in k}

        #count number of times word k[i] appears in other topics
        for i in range(len(k)):
            word = k[i]

            for topic_words in self.topics:
                if word in topic_words:

                    count[word] += 1
                    continue
        
        return count

    def form_cnt_lookup_table(self):

        for k in range(self.K):

            topic_words = self.topics[k]
            cnt_k = self.compute_cnt_per_topic(topic_words)
            
            self.cnt_lookup[k] = cnt_k
        
        print('Initialized cnt(i,k) for all words across all topics')

    def compute_topic_uniquess(self):

        self.form_cnt_lookup_table()

        for k in range(self.K):
            
            lookup = self.cnt_lookup[k]
            cnt_values = lookup.values()
            self.topic_uniqueness[k] = (1/len(lookup)) * sum([1/value for value in cnt_values])

        print('Computed Topic Uniqueness for all topics')

    def compute_average_topic_uniqueness(self):
        return sum(self.topic_uniqueness.values())/len(self.topic_uniqueness)

    def get_output(self):
        
        topic_uniqueness_for_all_topics = {}

        self.compute_topic_uniquess()

        for k in range(self.K):
            topic_uniqueness_for_all_topics[' '.join(self.topics[k])] = self.topic_uniqueness[k]

        return topic_uniqueness_for_all_topics