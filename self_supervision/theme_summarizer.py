import re
import nltk

import gensim
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from nltk.corpus import stopwords
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TF_IDF_Words:
    """
        Given a list of sentences, find the most important words using the TF-IDF method. The words can be unigrams, bigrams, or trigrams.

        Use the get_words() to get important words

        input : data <list>, bigrams <boolean>, trigrams <boolean>
                - data : list of strings you want to find important words for
                - bigrams : Set to true if you want important bigrams. Set to False by default.
                - trigrams : Set to true if you want important trigrams. Set to False by default.
        
        output : important_words <list>
    """
    
    def __init__(self, data, bigrams=False, trigrams=False):
        
        self.data = data
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.important_words = None
        self.stop_words = set(stopwords.words('english'))
        
        your_list = ['i','am','them','the','we','it','also','if', 'even','he','my','your','in',
                     'her','you','and','so','was','lot']
        
        for word in your_list:
            self.stop_words.add(word)
        
    def remove_string_special_characters(self, s):
      
        # removes special characters with ' '
        stripped = re.sub('[^a-zA-z\s]', '', s)
        stripped = re.sub('_', '', stripped)

        # Change any white space to one space
        stripped = re.sub('\s+', ' ', stripped)

        # Remove start and end white spaces
        stripped = stripped.strip()
        if stripped != '':
            return stripped.lower()
            
    def remove_stopwords(self, data):
            
        clean_data = [None]*len(data)
        idx = 0
        for line in data:
            try:
                clean_data[idx] = ' '.join([x.lower() for x in nltk.word_tokenize(line) if x.lower() not in self.stop_words])
                idx = idx+1
            except:
                clean_data[idx] = 'EMPTY'
                idx = idx+1
        
        return clean_data
            
    def n_gram_tf_idf(self, ngram, data):
        
        """
            Computes TF-IDF for ngrams -- not just unigrams but also for bigrams/trigrams.
        """
        
        try:
    
            vectorizer = CountVectorizer(ngram_range = (1, ngram))
            X1 = vectorizer.fit_transform(data) 
            features = (vectorizer.get_feature_names())

            # Applying TFIDF
            vectorizer = TfidfVectorizer(ngram_range = (1, ngram))
            X2 = vectorizer.fit_transform(data)
            scores = (X2.toarray().astype('float16'))
            
        except:
            """
                If you run out of memory for a task -- use only the most frequent words, limit max number of words
            """
            
            vectorizer = CountVectorizer(max_features = int(0.1*(len(data))), ngram_range = (1, ngram))
            X1 = vectorizer.fit_transform(data)
            features = (vectorizer.get_feature_names())
            
            vectorizer = TfidfVectorizer(max_features = int(0.1*(len(data))), ngram_range = (1, ngram))
            X2 = vectorizer.fit_transform(data)
            scores = (X2.toarray().astype('float16'))

        # Getting top ranking features
        sums = X2.sum(axis = 0)
        data1 = []
        for col, term in enumerate(features):
            data1.append( (term, sums[0,col] ))
        ranking = pd.DataFrame(data1, columns = ['term','rank'])
        words = (ranking.sort_values('rank', ascending = False))

        return words
    
    def get_words(self):
        
        """
            Returns unigrams/bigrams/trigrams in descending order of importance.
        """
        
        self.data = [self.remove_string_special_characters(s) for s in self.data]
        self.data = self.remove_stopwords(self.data)
        
        if self.bigrams:
            self.important_words = self.n_gram_tf_idf(ngram=2, data=self.data)
            self.important_words = self.important_words['term'].tolist()
            
        elif self.trigrams:
            self.important_words = self.n_gram_tf_idf(ngram=3, data=self.data)
            self.important_words = self.important_words['term'].tolist()

        else:
            self.important_words = self.n_gram_tf_idf(ngram=1, data=self.data)
            self.important_words = self.important_words['term'].tolist()
            
        return self.important_words

class LDA:

    """
    
    Class to get tokens using Latent Dirichlet Allocation.

    Get LDA words by running LDA(docs=docs, topn=topn).get_output() to get important words according to LDA.
        - docs (list): list of documents you want to run LDA on
        - topn (int): number of top aspects you want from the LDA output. Default is set to 3
    
    """

    def __init__(self, docs:list, topn=3, num_topics=4):

        self.docs = docs
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.num_topics = num_topics
        self.topn = topn
        self.dictionary = None
        self.lda_model = None
        self.bow_corpus = None
    
    def _get_wordnet_pos(self, word:str):
        
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    
    def _word_lemmatizer(self, text:str):
        
        text = self.lemmatizer.lemmatize(text, self._get_wordnet_pos(text))
        return text

    def _preprocess(self, text:str):
        
        text = re.sub(r'[\W ]+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = text.lower()
        
        tokens = [word for word in word_tokenize(text)]
        tokens = [word for word in tokens if len(word) >= 3]
        tokens = [self._word_lemmatizer(w) for w in tokens]
        tokens = [s for s in tokens if s not in self.stop_words]
        
        return tokens
    
    def _process_data(self):
        
        clean_data = []
        for text in self.docs:
            clean_data.append(self._preprocess(text))
            
        return clean_data
    
    def _get_dictonary(self, processed_docs):
        
        dictionary = gensim.corpora.Dictionary(processed_docs)
        
        self.dictionary = dictionary
        return dictionary
    
    def _get_bow_corpus(self, processed_docs):
        
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        self.bow_corpus = bow_corpus
        
    def _get_model(self, num_passes=10, num_workers=1):
        
        try:
            lda_model = gensim.models.LdaMulticore(self.bow_corpus, num_topics = self.num_topics, id2word = self.dictionary, passes = num_passes, workers = num_workers) 
        except RuntimeError as e:
            print('Multicore processing error -- using standard single core LDA ...',e)
            lda_model = gensim.models.LdaModel(self.bow_corpus, num_topics = self.num_topics, id2word = self.dictionary, passes = num_passes)

        self.lda_model = lda_model
        
    
    def get_topics(self):
        
        output = {}
        for idx in range(self.num_topics):

            word_list = self.lda_model.show_topic(idx)
            word_list = [word for word, _ in word_list]
            output['{}'.format(idx)] = word_list
            
        return output
    
    def get_output(self):
        """
        Get topic words according to LDA

        Returns:
            all_words (list) : a list of all important words as identified by LDA

        """
        clean_data = self._process_data()
        self._get_dictonary(clean_data)
        self._get_bow_corpus(clean_data)
        self._get_model()

        output = self.get_topics()
        all_words = [v for _,v in output.items()]
        all_words = [arr for sublist in all_words for arr in sublist]

        return all_words

class TopicWords:

    def __init__(self, n=None, bigrams=True, trigrams=True):

        self.topics = {}
        self.n = n

        self.bigrams = bigrams
        self.trigrams = trigrams

        if self.n is None:
            self.n = 20

    def split_lda_to_tfidf_ratio(self):
        """
        for any number of topic words, LDA terms : TFIDF terms should be 1:3

        Returns:
            num_lda_words (int): number of terms that should be taken from the LDA output
            num_tfidf_words (int): number of terms that should be taken from the TFIDF output
        """
        
        ratio = self.n/4
        num_lda_words = int(ratio*1)
        num_tfidf_words = self.n - num_lda_words

        return num_lda_words, num_tfidf_words

    def form_topic_words(self, data:list):
        """
        Forms words that summarize the topic using TF-IDF and LDA

        Args:
            data (list): list of strings to form topic words

        Returns:
            (list): ngrams and common words used to summarize the topic
        """
        
        #get number of terms to take from LDA and TFIDF
        num_lda_words, num_tfidf_words = self.split_lda_to_tfidf_ratio()
        
        #top ngrams using TF-IDF
        important_words = TF_IDF_Words(data=data, bigrams=self.bigrams, trigrams=self.trigrams).get_words()[:num_tfidf_words]
        commonly_associated_words = LDA(docs=data).get_output()

        #top common words using LDA
        commonly_associated_words = Counter(commonly_associated_words).most_common(num_lda_words)
        commonly_associated_words = [word for (word, _) in commonly_associated_words]

        topic_words = list(set(important_words + commonly_associated_words))

        return topic_words