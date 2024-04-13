import numpy as np
import pandas as pd
import nltk
import string
import re
import inflect
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class Text_Pipeline:
    """
    A class used to represent the Data Pipeline

    ...

    Attributes
    ----------
    _data_folder : str
        a string used to store the folder for the input and intermediate transformed data
    _source : str
        the name of the source file
    source_df : df
        the dataframe representing the source data
    transformed_df : df
        the dataframe representing the transformed data

    Methods
    -------
    preprocess()
        Perform preprocessing on a series which includes trimming whitespace, expansion, removal of stop words, removal of punctuation, 
        converting numbers, lemmatization
    encode()
        Encodes the text into numeric form

    """
    def __init__(self, number_handling='REMOVE'):
        """ Initializes the Data Pipeline Class

        """
        # Initialize various tools
        self.number_handling = number_handling
        self.encoder_dict = {"BOW": wordnet.ADJ,
                            "TFIDF": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}


    def preprocess(self, series):
        """ Preprocesses the series of text

        Parameters
        ----------
        series : Pandas Series
            The series of text that needs preprocessing

        Returns
        -------
        processed_text
            The processed series
        """
        # Apply preprocessing on the entire series
        #return series.apply(Text_Pipeline.preprocess_text)
        return series.apply(lambda x: Text_Pipeline.preprocess_text(x, self.number_handling))

    def encode(self, series, encoding_type="TFIDF"):
        """ Encodes the series 

        Parameters
        ----------
        series : Pandas Series
            The series of text that needs encoding
        encoding_type : str
            The encoding type from "BOW", "TFIDF" and "WordToVec"

        Returns
        -------
        processed_text
            The processed series
        """
        

        # Get the encoder & fit the data to it 
        if (encoding_type == 'WordToVec'):
            # WORD2VEC ENCODING
            text_array = series.apply(word_tokenize)
            word2vec_model = Word2Vec(text_array, vector_size=10, window=5, min_count=1, workers=4)
            return word2vec_model.wv
        else:
            # Get the values
            text_array = series.values
            encoder = self.get_encoder(encoding_type)
            matrix = encoder.fit_transform(text_array)
            return (matrix, encoder.get_feature_names_out())
    
    def get_encoder(self, encoding_type):
        if (encoding_type == 'BOW'):
            return CountVectorizer()
        elif (encoding_type == 'TFIDF'):
            return TfidfVectorizer()

    @staticmethod
    def preprocess_text(input_text, number_handling):
        """ Preprocesses the text

        Parameters
        ----------
        input_text : str
            The text that needs preprocessing

        Returns
        -------
        processed_text
            The processed text
        """
        # Remove whitespace
        transformed_text = Text_Pipeline.remove_whitespace(input_text)

        # Make the text lower case
        transformed_text = transformed_text.lower()

        # Expand Contractions
        transformed_text = Text_Pipeline.expand_contractions(transformed_text)

        # Remove punctuation and convert numbers 
        transformed_text = Text_Pipeline.remove_punctuation(transformed_text)

        # Handle numbers
        if (number_handling == 'CONVERT'):
            transformed_text = Text_Pipeline.remove_punctuation(Text_Pipeline.convert_number(transformed_text))
        elif (number_handling == 'REMOVE'):
            transformed_text = Text_Pipeline.remove_punctuation(Text_Pipeline.remove_numbers(transformed_text))

        # Remove stop words
        filtered_words = Text_Pipeline.remove_stopwords(transformed_text)

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        transformed_text = ' '.join([lemmatizer.lemmatize(w, Text_Pipeline.get_wordnet_pos(w)) for w in filtered_words])
        return transformed_text
    
    @staticmethod
    def expand_contractions(input_text):
        """ Expands contractions such as you've or aren't

        Parameters
        ----------
        input_text : str
            The text that needs expansion

        Returns
        -------
        expanded_text
            The expanded text

        """
        # Creating an empty list
        expanded_words = []    
        for word in input_text.split():
            # using contractions.fix to expand the shortened words
            expanded_words.append(contractions.fix(word))   
    
        expanded_text = ' '.join(expanded_words)
        return expanded_text
    
    @staticmethod
    def remove_whitespace(input_text):
        """ Removes extra whitespace

        Parameters
        ----------
        input_text : str
            The text that needs trimming

        Returns
        -------
        trimmed_text
            The trimmed text

        """
        return  " ".join(input_text.split())

    @staticmethod
    def remove_punctuation(input_text):
        """ Removes punctuations from text

        Parameters
        ----------
        input_text : str
            The text that needs punctuations removed

        Returns
        -------
        transformed_text
            The text sans punctuation

        """
        translator = str.maketrans('', '', string.punctuation)
        return input_text.translate(translator)
    
    @staticmethod
    def remove_stopwords(input_text):
        """ Removes stop words from text

        Parameters
        ----------
        input_text : str
            The text that needs stop words removed

        Returns
        -------
        transformed_text
            The text sans stop words

        """
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(input_text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return filtered_text

    @staticmethod
    def remove_numbers(text):
        """ Removes numbers from text 

        Parameters
        ----------
        input_text : str
            The text that needs numbers removed

        Returns
        -------
        transformed_text
            The text sans numbers 

        """
        result = re.sub(r'\d+', '', text)
        return result
    
    @staticmethod
    def convert_number(input_text):
        """ Replaces numbers from text with their respective word representations

        Parameters
        ----------
        input_text : str
            The text that needs numbers transformed to word representation

        Returns
        -------
        transformed_text
            The text with numbers transformed to word representation

        """
        inflect_engine = inflect.engine()

        # split string into list of words
        temp_str = input_text.split()

        # initialise empty list
        new_string = []
    
        for word in temp_str:
            # if word is a digit, convert the digit
            # to numbers and append into the new_string list
            if word.isdigit():
                temp = inflect_engine.number_to_words(word)
                new_string.append(temp)
    
            # append the word as it is
            else:
                new_string.append(word)
    
        # join the words of new_string to form a string
        temp_str = ' '.join(new_string)
        return temp_str

    @staticmethod
    def get_wordnet_pos(word):
        """ Map POS tag to first character lemmatize() accepts

        Parameters
        ----------
        word : str
            The word that needs its Tag gleaned

        Returns
        -------
        tag
            The tag associated for the word

        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)