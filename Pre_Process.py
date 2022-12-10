import re
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Pre_Process: 
    """
    class that assists in text cleaning pre-processing for text analysis
    """    
  
    def __init__(self, X): # argument should be array of text entries
        """
        initializes Pre_Process object and downloads stopwords to use in later method
        Args:
            X: array of strings of text
        Returns:
            nothing
        """   
        self.texts = X
        self.check()
        nltk.download('stopwords')
        self.stopwords = stopwords.words('english')
            
    def check(self, X = None): # checking input validity for text pre-processing
        """
        Checks if input 'X' is valid for text pre-processing. If X is not given, default to what X was given when Pre_Process was first initialized as
        Args:
            X: array of strings of text
        Returns:
            nothing. But raises exceptions / text describing error of X input
        """   
        if X is None:
            X = self.texts
        if (len(X) < 1):
            raise ValueError("Input must be at least 1")
        elif (not isinstance(X[0], str)):
            raise TypeError("Input must be of type string")  

    def remove_special_char(self, X): # takes an argument X and removes special characters (e.g. '-', '(')
        """
        Removes the special characters from each text of X (e.g. '-') and replaces with a single space. First checks validitiy of X
        Args:
            X: array of strings of text
        Returns:
            X with special characters removed
        """   
        self.check(X)
        return [re.sub(r'\W', ' ', str(t)) for t in X]
         
    def remove_single_char(self, X): # takes an argument X and removes single characters (e.g. 'a', 'b')
        """
        Removes the single characters from each text of X (e.g. 'a') and replaces with a single space. First checks validity of X
        Args:
            X: array of strings of text
        Returns:
            X with single characters removed
        """  
        self.check(X)
        X = [re.sub(r'\^[a-zA-Z]\s+', ' ', t) for t in X]
        return [re.sub(r'\s+[a-zA-Z]\s+', ' ', t) for t in X] 
    
    def remove_mult_space(self, X): # takes an argument X and removes duplicate spacing (e.g. '   ')
        """
        Removes multiple spaces from each text of X (e.g. '   ') and replaces with a single space. First checks validity of X
        Args:
            X: array of strings of text
        Returns:
            X with multiple spaces removed
        """  
        self.check(X)
        return [re.sub(r'\s+', ' ', t, flags=re.I) for t in X] 
    
    def to_lower(self, X): # takes an argument X and makes letters lowercased (e.g. 'A' -> 'a')
        """
        Converts strings to lowercase letters for each text of X (e.g. 'A' -> 'a'). First checks validity of X
        Args:
            X: array of strings of text
        Returns:
            X in all lowercase
        """ 
        self.check(X)
        return [t.lower() for t in X] 
    
    def remove_stopwords(self, X, stopwords = None):
        """
        Removes the stop words from each text of X (e.g. 'a', 'is') and replaces with a single space. First checks validity of X
        Args:
            X: array of strings of text
            stopwords: list of stopwords, if none is provided, default as stopwords from nltk
        Returns:
            X with stopwords removed
        """  
        self.check(X)
        if stopwords is None:
            stopwords = self.stopwords
        return [' '.join([word for word in text.split() if word not in stopwords]) for text in self.to_lower(X)]
        
    def lamentize(self, X):
        """
        Groups like words together, lemmatizing them
        Args:
            X: array of strings of text
        Returns:
            X with like words lemmatized
        """ 
        self.check(X)
        return [' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split()]) for text in self.to_lower(X)]
        
    def do_all(self): # calls all the methods on the original text dataset provided
        """
        Performs all the methods of pre-processing displayed above
        Args:
            None
        Returns:
            X fully pre-processed, removed special characters, single characters, multiple spaces, and stopwords, and also lemmantized.
        """  
        self.texts = self.remove_special_char(self.texts)
        self.texts = self.remove_single_char(self.texts)
        self.texts = self.remove_mult_space(self.texts)
        self.texts = self.remove_stopwords(self.texts)
        self.texts = self.lamentize(self.texts)
        return self.texts