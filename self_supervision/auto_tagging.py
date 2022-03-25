import sys

system_paths = sys.path
if './' not in system_paths:
    sys.path.extend(['./', '../'])

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from self_supervision.relation_extractor import ObjectExtractor

from self_supervision.embedding_former import EmbeddingFormer

from self_supervision.grouping_methods import similar_object_detection, similar_object_communities

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class GroupData:

    def __init__(self, object_groups):
        
        self.themes = object_groups

        #make a list to store texts belonging to each theme with other important information
        self.text2theme_mapping = []

        #make a dictionary of {object : texts in which object appears} for all objects
        self.text2object_mapping = {}

        self.df = pd.DataFrame()

    def reverse_lookup(self, dictionary:dict, value:str):
        """
        Given a value and a dictionary, return the first key in the dictionary that contains the value.
        If the dictionary does not have any key that contains the value, returns an empty string

        Args:
            dictionary (dict): dictionary | hashmap of items
            value (str): element to search for in the dictionary

        Returns:
            (str): first key of the dictionary that contains the value
        """
        for key, val in dictionary.items():
            if value in val:
                return key

        #if value is not in any of the dictionary keys, return empty string
        return ''

    def text2object(self, texts, objects):

        self.df = pd.DataFrame({'text':texts, 'objects':objects})
        for row in self.df.itertuples():
            
            #iterate objects in each row
            for o in row.objects:
                
                #add text that belong to each object

                if o in self.text2object_mapping:
                    self.text2object_mapping[o].append(row.text)
                else:
                    self.text2object_mapping[o] = [row.text]

    def text2theme(self, texts, objects):

        #map each text sample to the object it contains
        self.text2object(texts=texts, objects=objects)

        #maintain index of groups
        group_idx = 0

        #maintain a mapping of theme and text samples belonging to the theme
        for object_group in self.themes:
            
            overall_theme = object_group[0]
            related_themes = object_group[1:]
            texts_in_theme = [self.text2object_mapping[related_object] for related_object in object_group]

            self.text2theme_mapping.append({'idx':group_idx,
                                            'theme':overall_theme,
                                            'related_themes':related_themes,
                                            'texts':texts_in_theme})
            
            group_idx += 1

        #map the themes each data sample has
        theme_dict = {object_group[0] : object_group for object_group in self.themes}
        for k,v in theme_dict.items():
            print('{} --> {} ...'.format(k, v[:3]))

        themes_of_all_texts = []

        for row in self.df.itertuples():

            #get a set of unique themes by using reverse lookup of every object in a text sample
            themes_per_data_sample = set([self.reverse_lookup(dictionary=theme_dict, value=object) for object in row.objects])

            #if there is no overall theme for a sample, empty string is returned -- remove that
            themes_per_data_sample = [theme for theme in themes_per_data_sample if theme != '']

            #add these themes
            themes_of_all_texts.append(themes_per_data_sample)

        self.df['theme'] = themes_of_all_texts

        #faster one liner way of doing the same -- but less readability
        #reverse lookup for theme for EACH object of text samples -- if there is no overall theme, an empty string is returned; remove that
        # self.df['theme'] = self.df['objects'].apply(lambda x : list(set([t for t in [self.reverse_lookup(dictionary=theme_dict, value=ob) for ob in x] if t != ''])))

        return self.df

class SimilarRelationExtractor:

    def __init__(self, texts, embedding_model=embedding_model):

        self.texts = texts
        self.embedding_model = embedding_model

        self.df = pd.DataFrame()
        
        self.all_objects = []
        self.unique_objects = []
        
        self.object_embeddings = []
        self.object_groups = []

        self.modified_text_embeddings = []

    def form_dataframe_from_texts(self):
        self.df = pd.DataFrame({'text':self.texts})
    
    def extract_objects_from_texts(self):

        self.form_dataframe_from_texts()
        object_extractor =  ObjectExtractor()

        #create a column to tag all text samples with objects
        self.all_objects = object_extractor.extract_objects(self.texts, extend=False)
        self.df['objects'] = self.all_objects

        #flatten all objects
        self.unique_objects = [arr for sublist in self.all_objects for arr in sublist]
        
        #store the unique objects from the set of all objects
        self.unique_objects = list(set(self.unique_objects))

        print('{} unique objects from {} text samples'.format(len(self.unique_objects), len(self.texts)))

    def form_object_embeddings(self):

        self.extract_objects_from_texts()
        self.object_embeddings = self.embedding_model.encode(self.unique_objects)
        print('Formed embeddings for all unique objects ...')

    def form_augmented_text_embeddings(self):
        
        self.modified_text_embeddings = EmbeddingFormer(embedding_model = self.embedding_model,
                                                        augment = {'use_objects':True}
                                                        ).form_embeddings()
        print('Formed embeddings for all modified texts ...')

    def group_objects_by_similarity(self, threshold, min_group_size, mode='objects'):

        if mode == 'objects':
            unique_objects = self.unique_objects
            embeddings = self.object_embeddings

        elif mode == 'texts':
            unique_objects = self.texts
            embeddings = self.modified_text_embeddings

        self.object_groups = similar_object_detection(unique_objects = unique_objects,
                                                      embeddings = embeddings,
                                                      threshold = threshold,
                                                      min_group_size = min_group_size)

    def group_objects_by_clustering(self, threshold, min_group_size, mode='objects'):
        
        if mode == 'objects':
            unique_objects = self.unique_objects
            embeddings = self.object_embeddings

        elif mode == 'texts':
            unique_objects = self.texts
            embeddings = self.modified_text_embeddings

        self.object_groups = similar_object_communities(unique_objects = unique_objects,
                                                        embeddings = embeddings,
                                                        threshold = threshold,
                                                        min_community_size = min_group_size)

    def auto_tag_data(self, similarity_threshold=0.75, method='embedding_similarity', mode='objects', min_group_size=2):
        """
        Method to combine similar objects. Used to reduce noise by combining very similar objects together.
        Embeddings are formed from all unqiue objects using a SentenceTransformer model

        Args:
            similarity_threshold (float, optional): objects should be combined if their similarity is >= this value. Similarity is cosine_similarity(embedding(object_1), embedding(object_2)). Defaults to 0.75.
            method (str, optional): method used to combine objects. Defaults to 'embedding_similarity'.
                                    supported modes:
                                    'embedding_similarity' : forms a object embeddings and their similarity matrix, groups similar objects together
                                    'fast_clustering' : uses sentence_transformers' community detection implementation to cluster objects together
            mode (str, optional): mode used to combine texts. Defaults to 'objects'
                                  supported modes:
                                  'objects' : forms objects out of texts and uses these objects for clustering
                                  'texts' : augments texts with objects and uses these texts for clustering
            min_group_size(int, optional): minimum group size for it to be considered a valid object grouping. Defaults to 2.
        """

        if mode == 'objects':
            self.form_object_embeddings()

        elif mode == 'texts':
            self.form_augmented_text_embeddings()

        #group similar objects together
        if method == 'embedding_similarity':
            self.group_objects_by_similarity(threshold=similarity_threshold, min_group_size=min_group_size, mode=mode)

        elif method == 'fast_clustering':
            self.group_objects_by_clustering(threshold=similarity_threshold, min_group_size=min_group_size, mode=mode)

        print('Formed {} Groupings :\n\n'.format(len(self.object_groups)))

        #autotag data by theme -- each theme is an object grouping
        group_by_theme = GroupData(object_groups = self.object_groups)
        data_samples_with_themes = group_by_theme.text2theme(texts = self.texts, objects = self.all_objects)

        return data_samples_with_themes