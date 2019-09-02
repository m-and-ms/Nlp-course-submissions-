import nltk.corpus
from nltk import pos_tag
from nltk.corpus import *
from nltk.corpus import treebank
import numpy as np 
import pandas as pd



from subprocess import check_output
import pandas as pd
import os
from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import operator
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split  
import io
import requests
from nltk import pos_tag
from nltk import word_tokenize, pos_tag, pos_tag_sents
from bs4 import BeautifulSoup as bs
import requests
from pytube import YouTube
from selenium import webdriver
from sklearn.naive_bayes import GaussianNB

import time
from Naked.toolshed.shell import execute_js, muterun_js
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('universal_tagset')
nltk.download('treebank')


nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup 
import requests
from pytube import YouTube
#df.apply(lambda x: pd.lib.infer_dtype(x.values))
import time
from selenium.webdriver import Chrome
from contextlib import closing
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
def yscrap():
    comments=[]    
    chrome_options = Options()  
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(executable_path='/home/bora3i/Downloads/chromedriver')
    wait = WebDriverWait(driver,10)
    driver.get("https://www.youtube.com/watch?v=kffacxfA7G4")

    for item in range(20): #by increasing the highest range you can get more content
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
        time.sleep(3)

    for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text"))):
##        print(comment.text)
        comments.append(comment.text)
    print (comments)
    return(comments) 
def clean_comms(comms):
    coms_df=pd.DataFrame(comms)
    coms_df = coms_df.rename_axis('review', axis=1)

    print(coms_df.head())
    
def scrap():
    base = "https://www.youtube.com/results?search_query="
    qstring = "boddingtons+advert"
    r = requests.get(base+qstring)
    page = r.text
##    print(page)

    soup=bs(page,'html.parser')
    vids = soup.findAll(attrs={'class':'yt-uix-tile-link'})
    videolist=[]
    for v in vids:
        tmp = 'https://www.youtube.com' + v['href']
        videolist.append(tmp)
    print(videolist)


def get_coms():
    driver = webdriver.Chrome(executable_path='/home/bora3i/Downloads/chromedriver')

    driver.get('https://www.youtube.com/watch?v=iFPMz36std4')

    driver.execute_script('window.scrollTo(1, 500);')

 
    time.sleep(20)

    driver.execute_script('window.scrollTo(1, 3000);')



    comment_div=driver.find_element_by_xpath('//*[@id="contents"]')
    comments=comment_div.find_elements_by_xpath('//*[@id="content-text"]')
    for comment in comments:
       
        print(comment.text)


    
def low(df):
    
        
    df['review'] = df['review'].str.lower()
    return (df)
def tok(df):
    df['review'].apply(lambda x: [ nltk.word_tokenize(item) for item in x] )
    return (df)

def clean_html(txt):
  
    cleanr = re.compile('<.*?>')
    txt = re.sub(cleanr, '', txt)
    return (txt)

def call_clean(df):

    df['review'].apply(clean_html,1)
    return (df)

def restop(df):
 
    stop_words = set(stopwords.words('english'))

        
    
    df['review'].apply(lambda x: [item  for item in x if item not in stop_words])
        
        

 
    
    
#    for sentence in df:
#      for idx,rw in df.iterrows():
#        sentence = [w for w in sentence.lower().split() if w not in stop_words]
#        sentence = ' '.join(sentence)
#        df.loc[idx] = sentence
    return(df)








def re_put(txt):
   

  

        clean_text = re.compile('<.*?>,"')
        txt = re.sub(r'\d+', '', txt)
        txt = re.sub(clean_text, '', txt)
        txt = re.sub("'", '', txt)
        txt = re.sub(r'\W+', ' ', txt)
        txt = txt.replace('_', '')
##        df.loc[:,column].apply(lambda x: [item for item in x ])
        return(txt)
def rem_p(df):
    df['review'].apply(lambda x: x.translate(None, string.punctuation))
def call(df):

        
    df['review'].apply(re_put,1)
    return (df)

def lemtize(df):

    lems = WordNetLemmatizer()


##    stemm_w = []
   
##        word_tk = word_tokenize(column)
##        for word in word_tk:
##            stemm_w.append(lems.lemmatize(word))
##        column = ' '.join(stemm_w)
    df['review'].apply(lambda x:  ' '.join([lems.lemmatize(word) for word in x.split() ]))
    return(df)




def stemma(df):

    stem = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)
##
##    count = 0
##    stemed_wr = []
##    for sent in df:
##        w_tk = word_tokenize(sent)
##        for word in w_tk:
##            stemed_wr.append(stem.stem(word))
##        sent = ' '.join(stemed_wr)
##        df.iloc[count] = sent
##        count+=1
##        stemmed = []
    df['review'].apply(lambda x: ' '.join([stem.stem(word) for word in x.split() ]))
    return(df)
def fit_vec(df):
    count_vect = CountVectorizer(analyzer='word')
    X_train_counts = count_vect.fit_transform(df['review'])
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    count_vect.vocabulary_.get(u'algorithm')
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts).toarray()
    print(X_train_tf.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0) 
    clf =  RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))


def fit_vecN(df):
    count_vect = CountVectorizer(analyzer='word')
    X_train_counts = count_vect.fit_transform(df['review'])
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    count_vect.vocabulary_.get(u'algorithm')
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0) 
    
    print(X_train_tf.shape)
    clf =  GaussianNB()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))
def fit_vecNrf(df):
    
    # GB and tfidf vectorizer 
    vectorizer = TfidfVectorizer
    X_train_counts = vectorizer.fit_transform(df['review'])
    X_train_counts.toarray()
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X_train_counts, y, test_size=0.2, random_state=0) 
    X_train=X_train.toarray()
    print(X_train_tf.shape)
    clf =  GaussianNB()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))    
def fit_vectfrf(df):
    # USED tf VECTORIZER and rf 
    vectorizer = TfidfVectorizer
    X_train_counts = vectorizer.fit_transform(df['review'])
    X_train_counts.toarray()
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X_train_counts, y, test_size=0.2, random_state=0) 
    print(X_train_tf.shape)
    clf =  RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))

    
def cleantxt(text):
    stemmer = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)
    lems = WordNetLemmatizer()

    no_html = BeautifulSoup(text, "html.parser").get_text()
    tokens=[ x for x in word_tokenize(no_html) if x]

    tokens = [re.sub(r'[^\w\s]', '', x)  for x in tokens]
    tokens = [re.sub(r'\s+[a-zA-Z]\s+', '', x)  for x in tokens]
    tokens = [re.sub(r'\s+', '', x)  for x in tokens]
    tokens = [re.sub(r'^b\s+', '', x)  for x in tokens]
    tokens = [re.sub(r'\d+', '', x)  for x in tokens]








    tokens=[ stemmer.stem(x) for x in tokens if len(x)<=3 and x not in stopwords.words('english') ] 

    tokens= [lems.lemmatize(x) for x in tokens]
    tokens=[ x for x in tokens if x.lower()]
    return (" ".join(tokens))

        

def main3():
    #### REMOVING STOPS AND NAIVE BAYES CLASS
    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)# using removal stop
    fit_vecN(rev_lab_non_stop)
    
def main4():
    #### NOT REMOVING STOPS AND RANDFO CLASS
    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    fit_vec(rev_lab)
def main5():
    ##removing stops and RF CLASSIFIER
    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)

    rev_lab_non_stop=restop(rev_lab)
    fit_vec(rev_lab_non_stop)

    
def main6():
    #### NOT REMOVING STOPS AND NAIVE BAYES CLASS

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop_c=call(rev_lab)

#### didnt use stop words removal    
    fit_vecN(rev_lab_non_stop_c)



    
def main7():
####  REMOVING STOPS M LEMTAIZATION AND NAIVE BAYES CLASS

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)
    
    rev_lab_non_stop_c=call(rev_lab_non_stop)


    lem_rev_lab=lemtize(rev_lab_non_stop_c)
    


    fit_vecN(lem_rev_lab)       
def main8():
####  REMOVING STOPS M LEMTAIZATION AND RF CLASS

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)
    rev_lab_non_stop_c=call(rev_lab_non_stop)


    lem_rev_lab=lemtize(rev_lab_non_stop_c)
    
  
    fit_vec(lem_rev_lab)
def main9():
####  REMOVING STOPS M stemming AND NB CLASS

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)
    rev_lab_non_stop_c=call(rev_lab_non_stop)


    stem_rev_lab=stemma(rev_lab_non_stop_c)    

        
    fit_vecN(stem_rev_lab)
def main10():
####  REMOVING STOPS M stemming AND RF CLASS

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:50000,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)
    rev_lab_non_stop_c=call(rev_lab_non_stop)


    stem_rev_lab=stemma(rev_lab_non_stop_c)    

  
    fit_vec(stem_rev_lab)    
def main11():
####  REMOVING STOPS M stemming AND NB CLASS ,tf VEC

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)
    rev_lab_non_stop_c=call(rev_lab_non_stop)


    stem_rev_lab=stemma(rev_lab_non_stop_c)    

    
    fit_vecNrf(stem_rev_lab)
def main12():
####  REMOVING STOPS M stemming AND RF CLASS ,tf VEC

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)    
    rev_lab_non_stop_c=call(rev_lab_non_stop)


    stem_rev_lab=stemma(rev_lab_non_stop_c)    

      
    fit_vectfrf(stem_rev_lab)
def main13():
    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:50000,['type','review','label','file']]

    rev_lab=df2[['review','label']]
##    rev_lab=low(rev_lab)
##    rev_lab=tok(rev_lab)
##    rev_lab=call_clean(rev_lab)
##    
##    rev_lab_non_stop_c=call(rev_lab)
##
##
##    stem_rev_lab=stemma(rev_lab_non_stop_c)
####
####    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
####    df=pd.DataFrame(df)
####
####    df2=df.loc[1:10000,['type','review','label','file']]
####
##    rev_lab["review"]=rev_lab["review"].str.findall(r'\b\w[\w-]*\b') 


    cleaned=[]

    for rw in rev_lab['review']:
        rw=cleantxt(rw)
        
        cleaned.append(rw)
    return (cleaned)

   
def get_main13():
    positive_words=['awesome','good','nice','super','fun','delightful','better','fantastic','love','well','lovely','great','wow']
    negative_words=['awful','lame','horrible','bad','not nice','worse','terriable','hate','dislike','unliked','not well']    
    cleaned= main13()
  
    print(cleaned)
##    cleaned = re.findall(r'\b\w[\w-]*\b', cleaned.lower())
##    print(cleaned)
    

    numPosWords = 0
    for sent in cleaned:
        words = re.findall(r'\b\w[\w-]*\b', sent.lower())
        gathered=" ".join(words)
       
        for w in words:
            
  


    
            if w in positive_words:
                numPosWords += 1
                print(numPosWords)
              
    numNegWords = 0
    for sent in cleaned:
        words = re.findall(r'\b\w[\w-]*\b', sent.lower())
        gathered=" ".join(words)
        for w in words:
        
            if w in negative_words:
                numNegWords += 1
                
    
    numWords = len(cleaned)
    percntPos = numPosWords / numWords
    percntNeg = numNegWords / numWords
    print("Positive: " + "{:.0%}".format(percntPos) + "  Negative: " + "{:.0%}".format(percntNeg))

def add_not(df):
    
    positive_words=['awesome','good','nice','super','fun','delightful','better','fantastic','love','well','lovely','great','wow']
    negative_words=['awful','lame','horrible','bad','not nice','worse','terriable','hate','dislike','unliked','not well','not']    
##    cleaned= main13()
##    cleaned = re.findall(r'\b\w[\w-]*\b', cleaned.lower())
##    print(cleaned)
    


    for i,sent in enumerate(df['review']):
      
##        words = re.findall(r'\b\w[\w-]*\b', word.lower())
        words=nltk.word_tokenize(sent)
##        print(words)
        
        add='NOT_'
        neg_w='not'
        neg2='nt'
        

            
        for i,word in enumerate(words):
            if ((word == neg_w) or (word == neg2)  ):
                words[i]='NOT_'
                j=i+1
                while(j<=len(words)-1):
                    if (words[j]=='nt'):
                        words[j]='NOT_'
                    else:    
                        words[j]=words[j]+'NOT_'
                    j+=1
                     
                     
        gathered=' '.join(words)
        print(gathered)
        df.loc[i,'review']=gathered
                

                    
       
             
       
                     
        print(df)     
        return(df)


def thenot():

    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)

    df2=df.loc[1:,['type','review','label','file']]

    rev_lab=df2[['review','label']]
    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
    rev_lab_non_stop=restop(rev_lab)    
    rev_lab_non_stop_c=call(rev_lab_non_stop)


    stem_rev_lab=stemma(rev_lab_non_stop_c)    

    notadd=add_not(stem_rev_lab)  
    fit_vecN(notadd)
thenot()    
    

###i wasnt able to compile all the sample so i can never comment on the over all acuuracy  of course removing stops does decrease the dimensionality
## and enhance the performance , cleaning and stemming as well
###### Naive Bayes calssifier showed better vresults
####   countt vectorizer gives a slight higher acc  and definetly tf-idf is better
### run 12 main to do the linrealy combined ecpiruments
##get_main13 to count no of postives and negatives in dataframe

##thenot adds the not feature i chose to fit it on NB With cont vectorizer ana it usually gives better accuracy 
## tried ll possible solutions 
