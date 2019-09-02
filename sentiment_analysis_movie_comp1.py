import csv 
import pandas as pd 

import pickle


comments =pd.read_csv('movie_data.csv')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split




from sklearn.feature_extraction.text import TfidfVectorizer 



import re, string, unicodedata
import nltk
nltk.download('stopwords')

import nltk 

from nltk.tokenize import word_tokenize, sent_tokenize 

nltk.download('averaged_perceptron_tagger')

from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

nltk.download('wordnet')

def stemmer(new_word):
    stemmer = SnowballStemmer("english")
   
   
    stem = stemmer.stem(new_word)
              
    return stem



def lemmitizer(word):
 wordnet_lemmatizer = WordNetLemmatizer()
 newword= wordnet_lemmatizer.lemmatize(word)
 return newword

def clean(text  ,LS ,Rs=True,Not3=False,Notall=False):
    indexes=[]
    no_html = BeautifulSoup(text, "html.parser").get_text()
    
    tokens=[ x for x in word_tokenize(no_html) if x]
    
    tokens = [re.sub(r'[^\w\s]', '', x)  for x in tokens]
    
    tokens = [re.sub(r'\s+', '', x)  for x in tokens]
    if Rs :
     tokens = [x for x in tokens if x not in stopwords.words('english')]
    if LS=='S':
      tokens=[ stemmer(x) for x in tokens  ] 
    elif LS=='L':
     tokens= [lemmitizer(x) for x in tokens]	
    else :
      tokens=[ stemmer(x) for x in tokens ] 
      tokens= [lemmitizer(x) for x in tokens]
    
    if Notall :

      indexes=[i for i,word in enumerate(tokens) if word in ['not' ,'Not' ,'nt']]
      
      for j in indexes :
        if j :
          notword_set=['not'+"_"+x for x in tokens[j+1:]]
          tokens=tokens[:j-1]+notword_set
        else :
          tokens=tokens
        print (indexes," ".join(tokens))
    if Not3 :

      indexes=[i for i,word in enumerate(tokens) if word in ['not' ,'Not' ,'nt']]
      
      for j in indexes :
        if j :
          notword_set=['not'+"_"+x for x in tokens[j+1:j+4]]
          tokens=tokens[:j-1]+notword_set
        else :
          tokens=tokens
        print (indexes," ".join(tokens))



    
    return (" ".join(tokens))
  







def tfidf(X): 
  tfidfconverter = TfidfVectorizer( min_df=5)  
  X = tfidfconverter.fit_transform(X).toarray()  
  return X


from sklearn.feature_extraction.text import CountVectorizer 

def cv(X): 
  vectorizer = CountVectorizer(binary=True)  
  X = vectorizer.fit_transform(X)

  return X











#print(train_X)
#print(train_Y) 


#print(train_X)

def RF(train_X ,train_Y ,test_X,test_Y):

   classifier = RandomForestClassifier(n_estimators=50, random_state=0)  
   classifier.fit(train_X,train_Y)  

   pred = classifier.predict(test_X)  
   print(confusion_matrix(test_Y,pred))  
   print(classification_report(test_Y,pred))  
   print(accuracy_score(test_Y,pred))
   #with open('tfidf_text_classifier', 'wb') as picklefile:  
   #pickle.dump(classifier,picklefile)  



def bayes(train_X ,train_Y ,test_X,test_Y):

   classifier =GaussianNB()
   classifier.fit(train_X, train_Y)

   pred = classifier.predict(test_X)  
   print(confusion_matrix(test_Y,pred))  
   print(classification_report(test_Y,pred))  
   print(accuracy_score(test_Y,pred))









def main1():

    X=[clean(x,'S') for x in comments['review']]



    Y =comments['sentiment']
    X=cv(X)
    train_X ,test_X,train_Y , test_Y = train_test_split(X,Y, test_size = 0.2)
    RF(train_X ,train_Y ,test_X,test_Y)




def main2():

    X=[clean(x,'L') for x in comments['review']]



    Y =comments['sentiment']
    X=cv(X)
    train_X ,test_X,train_Y , test_Y = train_test_split(X,Y, test_size = 0.2)
    RF(train_X ,train_Y ,test_X,test_Y)


def main3():

    X=[clean(x,'LS' ,Rs=False) for x in comments['review']]



    Y =comments['sentiment']
    X=cv(X)
    train_X ,test_X,train_Y , test_Y = train_test_split(X,Y, test_size = 0.2)
    RF(train_X ,train_Y ,test_X,test_Y)



def main4():

    X=[clean(x,'LS') for x in comments['review']]



    Y =comments['sentiment']
    X=tfidf(X)
    train_X ,test_X,train_Y , test_Y = train_test_split(X,Y, test_size = 0.2)
    RF(train_X ,train_Y ,test_X,test_Y)



def main5():

    X=[clean(x,'LS') for x in comments['review']]



    Y =comments['sentiment']
    X=tfidf(X)
    train_X ,test_X,train_Y , test_Y = train_test_split(X,Y, test_size = 0.2)
    bayes(train_X ,train_Y ,test_X,test_Y)




def main6():

    X=[clean(x,'LS',Not3=True ,Rs=False) for x in comments['review']]
    



    Y =comments['sentiment'][:]
    X=cv(X)
    train_X ,test_X,train_Y , test_Y = train_test_split(X,Y, test_size = 0.2)
    RF(train_X ,train_Y ,test_X,test_Y)




def main7():

    X=[clean(x,'LS',Notall=True,Rs=False) for x in comments['review']]



    Y =comments['sentiment']
    X=cv(X)
    train_X ,test_X,train_Y , test_Y = train_test_split(X,Y, test_size = 0.2)
    RF(train_X ,train_Y ,test_X,test_Y)


if __name__ == "__main__":

    main1()
