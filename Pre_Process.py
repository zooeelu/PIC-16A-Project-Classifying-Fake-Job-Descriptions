import re
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Pre_Process: 
    
    def __init__(self, X): # argument should be array of text entries
          
        self.texts = X
        self.check()
        nltk.download('stopwords')
        self.stopwords = stopwords.words('english')
            
    def check(self, X = None): # checking input validity for text pre-processing
        if X is None:
            X = self.texts
        try:
            if (len(X) < 1):
                raise Exception("Input must be at least length 1")
            elif (not isinstance(X[0], str)):
                print("Input must be of type string") 
                raise Exception("Input must be of type string")  
        except:
            print("An error has occured from you input. Make sure it is of type string and a vector of at least length 1")

    def remove_special_char(self, X): # takes an argument X and removes special characters (e.g. '-', '(')
        self.check(X)
        return [re.sub(r'\W', ' ', str(t)) for t in X]
         
    def remove_single_char(self, X): # takes an argument X and removes single characters (e.g. 'a', 'b')
        self.check(X)
        X = [re.sub(r'\^[a-zA-Z]\s+', ' ', t) for t in X]
        return [re.sub(r'\s+[a-zA-Z]\s+', ' ', t) for t in X] 
    
    def remove_mult_space(self, X): # takes an argument X and removes duplicate spacing (e.g. '   ')
        self.check(X)
        return [re.sub(r'\s+', ' ', t, flags=re.I) for t in X] 
    
    def to_lower(self, X): # takes an argument X and makes letters lowercased (e.g. 'A' -> 'a')
        self.check(X)
        return [t.lower() for t in X] 
    
    def remove_stopwords(self, X, stopwords = None): # takes arguments stopwords and X and makes letters lowercased (e.g. 'A' -> 'a')
        self.check(X)
        if stopwords is None:
            stopwords = self.stopwords
        return [' '.join([word for word in text.split() if word not in stopwords]) for text in self.to_lower(X)]
        
    def lamentize(self, X):
        self.check(X)
        return [' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split()]) for text in self.to_lower(X)]
        
    def do_all(self): # calls all the methods on the original text dataset provided
        self.texts = self.remove_special_char(self.texts)
        self.texts = self.remove_single_char(self.texts)
        self.texts = self.remove_mult_space(self.texts)
        self.texts = self.remove_stopwords(self.texts)
        self.texts = self.lamentize(self.texts)
        return self.texts