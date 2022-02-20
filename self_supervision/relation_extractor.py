import sys

system_paths = sys.path
if './' not in system_paths:
    sys.path.extend(['./', '../'])

import re
import spacy

from preprocessing.constants import *

nlp = spacy.load('en_core_web_md')

class RelationExtractor:

    def __init__(self):
        self.noisy_precedence_tokens = ['is', 'was', 'be', 'with', 'to', 'from', 'on', 'in', 'as',
                            'of', 'are', 'has', 'had', 'have', 'into', 'so', 'by', 'how',
                            'for', 'with', 'like', 'while']

    def preprocess_text(self, texts) -> list:

        if isinstance(texts, str):
            texts = [texts]

        texts = [re.sub(" \d+ ", "multiple ", " "+text) for text in texts]
        return texts

    def extract_if_dep(self, doc, tags, exclusion_list=[]) -> str:

        dep_in_object_string = [token for token in doc if token.dep_ in tags]

        if exclusion_list != [] and dep_in_object_string:

            #if any terms of the exclusion list are in the extracted tags in object string, discard it
            result = set(dep_in_object_string).intersection(set(exclusion_list))
            if result != set():
                return ''

        #if there are no required dependencies in the object string, discard it
        if dep_in_object_string == []:
            return ''

        return doc.text

    def extract_if_tag(self, doc, tags, exclusion_list=[]) -> str:

        tag_in_object_string = [token for token in doc if token.pos_ in tags]

        if exclusion_list != [] and tag_in_object_string:
            
            #if any terms of the exclusion list are in the extracted tags in object string, discard it
            result = set(tag_in_object_string).intersection(set(exclusion_list))
            if result != set():
                return ''

        #if there are no tags in the object string, discard it
        if tag_in_object_string == []:
            return ''
        
        return doc.text

    def prune_objects(self, objects:list):

        #object strings that start with a words like 'is', 'was', 'to' etc. are duplicates or not very useful usually, remove them
        for object_string in objects:
            
            first_word = object_string.strip().split()[0]
            if first_word in self.noisy_precedence_tokens:
                objects.remove(object_string)

            #if object is only one word long, remove it
            elif len(object_string.split()) == 1:
                objects.remove(object_string)
                
        return objects

class ObjectExtractor(RelationExtractor):

    def __init__(self):

        super().__init__()
        self.object_tags = object_tags
        self.modifier_tags = modifier_tags

        self.noun_exclusion_list = exclusion_list

    def get_only_objects_with_noun(self, objects:list, nlp=nlp):

        for doc in nlp.pipe(objects):

            #extract object string only if it has any noun tag in it
            object_string = self.extract_if_tag(doc=doc, tags=noun_tags, exclusion_list=self.noun_exclusion_list)

            #if result of extraction is an empty string, no noun tag was found -- remove object string from the objects
            if object_string == '':
                objects.remove(doc.text)

        return objects

    def get_only_objects_with_modifier(self, objects:list, nlp=nlp):

        for doc in nlp.pipe(objects):

            #extract object string only if it has any modifier tag in it
            object_string = self.extract_if_tag(doc=doc, tags=self.modifier_tags)

            #if result of extraction is an empty string, no modifier tag was found -- remove object string from the objects
            if object_string == '':
                objects.remove(doc.text)
        
        #return only those objects that have a modifier in them
        return objects

    def extract_object(self, doc, extend=False, prune=False):

        #if you want more detailed objects, add the token.children to the object phrase as well
        if extend:
            objects = [sorted([(token, token.i), (token.head, token.head.i)]+[(t, t.i) for t in token.children],
                       key = lambda x : x[1]) for token in doc if token.dep_ in self.object_tags]

        #find word with object tag, find what it refers to using token.head, sort them in order of occurrence in the phrase
        else:
            objects = [sorted([(token, token.i), (token.head, token.head.i)],
                       key = lambda x : x[1]) for token in doc if token.dep_ in self.object_tags]

        #convert to string format
        objects = [' '.join(word.text for word, position in object_pair) for object_pair in objects]

        #remove objects that start with noisy verb/preposition tokens
        if prune:
            objects = self.prune_objects(objects)

        #remove objects that dont have a noun in them
        objects = self.get_only_objects_with_noun(objects)

        #remove objects that dont have a modifier in them
        objects = self.get_only_objects_with_modifier(objects)

        #remove empty strings from objects and convert to lowercase
        objects = [object_string.strip().lower() for object_string in objects if object_string != '']

        return objects

    def extract_objects(self, phrases, nlp=nlp, extend=False, prune=False):
        """
        Extracts objects from a list of phrases
        objects are identified using the spacy dependency parser. Visualize dependency parser : https://explosion.ai/demos/displacy

        for a list of dependencies identified as objects, refer preprocessing.constants.py

        Args:
            phrases (list): list of strings for which objects are extracted
            nlp (spacy.lang.en model, optional): spacy.lang.en model. Defaults to 'en_core_web_sm'.
            extend (bool, optional): use entire object string with reference to its children, instead of using just object string. Defaults to False.
            prune (bool, optional): remove objects that start with some verb/preposition tokens. Defaults to False.

        Returns:
            (list): list of list notation, where each list has object strings of its respective phrase
        """

        if isinstance(phrases, str):
            phrases = [phrases]

        phrases = self.preprocess_text(phrases)
        
        objects_in_all_phrases = []

        for doc in nlp.pipe(phrases):
            
            objects_in_phrase = self.extract_object(doc=doc, extend=extend, prune=prune)
            objects_in_all_phrases.append(objects_in_phrase)

        return objects_in_all_phrases

class SubjectExtractor(RelationExtractor):

    def __init__(self):

        super().__init__()
        self.subject_tags = subject_tags
        self.pronoun_exclusion_list = exclusion_list

    def get_only_subjects_without_personal_pronouns(self, subjects):
        
        subjects_without_personal_pronouns = []

        for subject in subjects:

            tokens_in_subject = subject.lower().split()
            tokens_in_subject = [token for token in tokens_in_subject if token not in self.pronoun_exclusion_list]
            tokens_in_subject = ' '.join(tokens_in_subject)

            subjects_without_personal_pronouns.append(tokens_in_subject)

        return subjects_without_personal_pronouns

    def extract_subject(self, doc, extend=False):

        #if you want more detailed subjects, add the token.children to the subject phrase as well
        if extend:
            subjects = [sorted([(token, token.i), (token.head, token.head.i)]+[(t, t.i) for t in token.children],
                       key = lambda x : x[1]) for token in doc if token.dep_ in self.subject_tags]

        #find word with subject tag, find what it refers to using token.head, sort them in order of occurrence in the phrase
        else:
            subjects = [sorted([(token, token.i), (token.head, token.head.i)],
                       key = lambda x : x[1]) for token in doc if token.dep_ in self.subject_tags]

        #convert to string format
        subjects = [' '.join(word.text for word, position in subject_pair) for subject_pair in subjects]

        #remove personal pronouns from the subjects
        subjects = self.get_only_subjects_without_personal_pronouns(subjects=subjects)

        #remove empty strings from subjects and convert to lowercase
        subjects = [subject_string.strip().lower() for subject_string in subjects if subject_string != '']

        return subjects
        
    def extract_subjects(self, phrases, nlp=nlp, extend=False):
        """
        Extracts subjects from a list of phrases
        objects are identified using the spacy dependency parser. Visualize dependency parser : https://explosion.ai/demos/displacy

        for a list of dependencies identified as subjects, refer preprocessing.constants.py

        Args:
            phrases (list): phrases (list): list of strings for which objects are extracted
            nlp (spacy.lang.en model, optional): spacy.lang.en model. Defaults to 'en_core_web_sm'.
            extend (bool, optional): use entire subject string with reference to its children, instead of using just object string. Defaults to False.

        Returns:
            (list): list of list notation, where each list has subject strings of its respective phrase
        """

        if isinstance(phrases, str):
            phrases = [phrases]

        phrases = self.preprocess_text(phrases)

        subjects_in_all_phrases = []

        for doc in nlp.pipe(phrases):

            subjects_in_phrase = self.extract_subject(doc=doc, extend=extend)
            subjects_in_all_phrases.append(subjects_in_phrase)

        return subjects_in_all_phrases