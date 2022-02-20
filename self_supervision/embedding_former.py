import sys

system_paths = sys.path
if './' not in system_paths:
    sys.path.extend(['./', '../'])

import numpy as np
from numba import jit

import gc

from sentence_transformers import SentenceTransformer

from self_supervision.relation_extractor import ObjectExtractor, SubjectExtractor, MentionedSubjectAndPredicateExtractor
from metrics.scoring_mechanisms import GloVeEmbeddings

class AverageVectorEmbedding:

    def __init__(self, word_embeddings, word_embedding_size=50):

        self.word_embeddings = word_embeddings
        self.embedding_shape = word_embedding_size

        print('Initialized word embeddings to form average vectors ...')

    def form_average_vector(self, text):

        words = text.split()
        average_vector = []

        for word in words:

            try:
                vector = self.word_embeddings[word]
            except:
                vector = np.zeros(self.embedding_shape)

            average_vector.append(vector)

        #form column wise average for all row of word embeddings
        average_vector = np.array(average_vector).mean(axis=0)
        
        return average_vector

    def form_average_vector_embeddings(self, texts):

        print('Forming average vectors from word embeddings ...\n\n')

        embeddings = [self.form_average_vector(text) for text in texts]

        return embeddings

    def encode(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        return self.form_average_vector_embeddings(texts=texts)

class EmbeddingFormer:
    """
    
    Forms embeddings for text samples

    run EmbeddingFormer(embedding_model=embedding_model, augment=augment).form_embeddings() to get output
    """

    def __init__(self,
                embedding_model:str='paraphrase-MiniLM-L6-v2',
                augment:dict={}):
        """
        Initialize embedding formation

        Uses sentence_transformers to form sentence embeddings, or a custom GloveEmbeddings() class to form word embeddings and AverageVectorEmbeddings() to form average sentence representation using the word embeddings

        Args:
            embedding_model (str, optional): type of sentence embedding model to use. Defaults to 'paraphrase-MiniLM-L6-v2'.
                                             available modes:
                                             - 'glove' : to use glove word embeddings and then convert them to average vector embeddings
                                             - any model card (str) supported by sentence_transformers

            augment (dict, optional): dictionary to control data augmentation. Defaults to {'use_objects':False, 'expand_objects':False, 'use_subjects':False, 'expand_subjects':False}.
                                      set the following keys to True to augment data:
                                      - 'use_objects' : runs ObjectExtractor().extract_objects(prune=True) to extract objects. Defaults to False
                                      - 'expand_objects' : runs ObjectExtractor().extract_objects(extend=True, prune=True) to extract objects. Defaults to False
                                      - 'use_subjects' : runs SubjectExtractor().extract_objects() to extract subjects. Defaults to False
                                      - 'expand_subjects' : runs SubjectExtractor().extract_objects(extend=True) to extract subjects. Defaults to False
        """

        self.embedding_model = embedding_model
        self.augmentation = {'use_objects':False, 'expand_objects':False, 'use_subjects':False, 'expand_subjects':False}
        self.augmentation.update(augment)

    def augment_data_with_subjects_and_objects(self, texts):
        
        extend_subjects = self.augmentation['expand_subjects']
        extend_objects = self.augmentation['expand_objects']

        subject_extractor = SubjectExtractor()
        object_extractor = ObjectExtractor()

        subject_predicate_extractor = MentionedSubjectAndPredicateExtractor(subject_extractor_instance=subject_extractor,
                                                                            object_extractor_instance=object_extractor)

        mentions_in_texts = subject_predicate_extractor.extract_entities(texts = texts,
                                                                         extend_subjects=extend_subjects,
                                                                         extend_objects=extend_objects)

        modified_data = ['{}. This text talks about {}'.format(data_sample, ', '.join(mention)) for data_sample, mention in zip(texts, mentions_in_texts)]

        print('Augmented data with subjects and objects : {}\n.....\n'.format(modified_data[:5]))

        #remove from memory
        del subject_extractor, object_extractor, mentions_in_texts
        gc.collect()

        return modified_data

    def augment_data_with_objects(self, texts):

        extend = self.augmentation['expand_objects']

        extractor = ObjectExtractor()
        objects_in_texts = extractor.extract_objects(phrases = texts, extend=extend, prune=True)
        
        modified_data = ['{}. This text talks about {}'.format(data_sample, ', '.join(objects_in_text)) for data_sample, objects_in_text in zip(texts, objects_in_texts)]

        print('Augmented data with objects : {}\n.....\n'.format(modified_data[:5]))

        #remove from memory
        del extractor, objects_in_texts
        gc.collect()

        return modified_data

    def augment_data_with_subjects(self, texts):
        
        extend = self.augmentation['expand_subjects']
        
        extractor = SubjectExtractor()
        subjects_in_texts = extractor.extract_subjects(phrases = texts, extend=extend)

        modified_data = ['{}. This text talks about {}'.format(data_sample, ', '.join(subjects_in_text)) for data_sample, subjects_in_text in zip(texts, subjects_in_texts)]

        print('Augmented data with subjects : {}\n.....\n'.format(modified_data[:5]))

        #remove from memory
        del extractor, subjects_in_texts
        gc.collect()

        return modified_data

    def form_sentence_embeddings(self, texts):

        sentence_embedding_model = SentenceTransformer(self.embedding_model)
        embeddings = sentence_embedding_model.encode(texts)

        return embeddings

    def form_average_embeddings(self, texts):

        embedding_model = GloVeEmbeddings()
        embedding_model.initialize_embeddings()

        word_embeddings, embedding_size = embedding_model.embeddings, embedding_model.embedding_size

        sentence_embedding_model = AverageVectorEmbedding(word_embeddings=word_embeddings, word_embedding_size=embedding_size)
        embeddings = sentence_embedding_model.encode(texts)

        del word_embeddings, embedding_size
        gc.collect()

        return embeddings

    def augment_data(self, texts):
        
        if isinstance(texts, str):
            texts = [texts]

        if self.augmentation['use_subjects'] and self.augmentation['use_objects']:
            return self.augment_data_with_subjects_and_objects(texts=texts)

        if self.augmentation['use_subjects']:
            return self.augment_data_with_subjects(texts=texts)

        if self.augmentation['use_objects']:
            return self.augment_data_with_objects(texts=texts)

        return texts

    def form_embeddings(self, texts):

        texts = self.augment_data(texts = texts)

        if self.embedding_model == 'glove':
            print('Using GloVe word embeddings ...')
            embeddings = self.form_average_embeddings(texts = texts)

        else:
            embeddings = self.form_sentence_embeddings(texts = texts)

        return embeddings