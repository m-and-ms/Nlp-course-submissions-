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
from nltk import pos_tag
from nltk import word_tokenize, pos_tag, pos_tag_sents
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('universal_tagset')
nltk.download('treebank')


nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
##import BeautifulSoup
#df.apply(lambda x: pd.lib.infer_dtype(x.values))
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
def pos_t(df):
##    df['review'].apply(word_tokenize)
##    df['review'].apply(word_tokenize).tolist()
##    pos_tag_sents( df['review'].apply(word_tokenize).tolist() )
    df['POS'] =  df['review'].str.split().map(pos_tag).tolist()

    return (df)
def fit_vec(df):
    count_vect = CountVectorizer(analyzer='word')
    X_train_counts = count_vect.fit_transform(df['review'])
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    count_vect.vocabulary_.get(u'algorithm')
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0) 
    clf =  RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))
def fit_vecpos(lpos,df):
    count_vect = CountVectorizer(analyzer='word')
    X_train_counts = count_vect.fit_transform(lpos)
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    count_vect.vocabulary_.get(u'algorithm')
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0) 
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








    tokens=[ stemmer.stem(x) for x in tokens if len(x)<=3 and x not in stopwords.words('english') ] 

    tokens= [lems.lemmatize(x) for x in tokens]
    tokens=[ x for x in tokens if x]
    words=nltk.pos_tag(tokens)
    pos_tokens =[ x[0]+'_'+x[1]  for x in  words ]
    


    return (" ".join(pos_tokens))
def fit_vecN(df):
    count_vect = CountVectorizer(analyzer='word')
    X_train_counts = count_vect.fit_transform(df['review'])
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    count_vect.vocabulary_.get(u'algorithm')
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0) 
    
    print(X_train_tf.shape)
    clf = MultinomialNB().fit(X_train_tf, y)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))
def fit_vecCOUNT(df):
    # USED COUNT VECTORIZER
    count_vect = CountVectorizer(analyzer='word')
    X_train_counts = count_vect.fit_transform(df['review'])
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    count_vect.vocabulary_.get(u'algorithm')
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape
    X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0) 
    
    print(X_train_tf.shape)
    clf = MultinomialNB().fit(X_train_tf, y)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))
def fit_vecCOUNT_RF(df):
    # USED COUNT VECTORIZER
    count_vect = CountVectorizer(analyzer='word')
    X_train_counts = count_vect.fit_transform(df['review'])
    print(X_train_counts.shape)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    count_vect.vocabulary_.get(u'algorithm')
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape
    X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0) 
    
    print(X_train_tf.shape)
    clf =  RandomForestClassifier(n_estimators=100, random_state=0)

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))
    
def main():
    tr=treebank.raw()
    tg=treebank.tagged_words()
    
##print([tag[0] for tag in treebank.tagged_words(tagset='universal')])
    
    tags=[tag[1] for tag in treebank.tagged_words(tagset='universal')]
  
    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)
##    print (df.head())

    df2=df.loc[1:,['type','review','label','file']]
##    print (df2.head())
##    rev=df.loc[1:,['review']]
##    lab=df.loc[1:,['label']]
##    print(rev.head())
##    print(lab.head())
    rev_lab=df2[['review','label']]


    rev_lab=low(rev_lab)
    rev_lab=tok(rev_lab)
    rev_lab=call_clean(rev_lab)
##    print(rev_lab.head())
##
    rev_lab_non_stop=restop(rev_lab)

    rev_lab_non_stop_c=call(rev_lab_non_stop)
##    print(rev_lab_non_stop_c.head())


    lem_rev_lab=lemtize(rev_lab_non_stop_c)
    

    stem_rev_lab=stemma(lem_rev_lab)
##
##    print(lem_rev_lab.head())
##
##    print(stem_rev_lab.head())
    fd=nltk.FreqDist(tags)
    Comm=fd.most_common()
    tagged_revs=pos_t(stem_rev_lab)
    print(tagged_revs.head())
    nwf_pos=[]
    nwt_pos=[]
    nw_pos=[]
    for rw in tagged_revs['POS']:
        
        for (k,v) in rw:
##            print(k)
##
            nw_pos.append(str(k)+'_'+str(v))
##        nwt_pos.append(nw_pos)
##        nw_pos=[]
##    nwf_pos.append(nwt_pos)
##    nwt_pos=[]
##    print(nwf_pos[1])
##    nw_pos=' '.join(nw_pos)

##    tagged_revs['newpos']=nwf_pos
##    print(tagged_revs.head())
            
    fit_vec(stem_rev_lab)
##    fit_vecpos(nw_pos[:stem_rev_lab.count()],stem_rev_lab)
##            print(type(itm))
##        print('0000000000000000000000000000000000000000000000000')
##        
    
def main2():
    df = pd.read_csv('imdb_master.csv', encoding = 'ISO-8859-1')
    df=pd.DataFrame(df)
##    print (df.head())

    df2=df.loc[1:,['type','review','label','file']]
##    print (df2.head())
##    rev=df.loc[1:,['review']]
##    lab=df.loc[1:,['label']]
##    print(rev.head())
##    print(lab.head())
    rev_lab=df2[['review','label']]
    cleaned=[]
    for rw in rev_lab['review']:
        cleaned.append(cleantxt(rw))
        
            
    fit_vec(cleaned)
       
    fit_vecpos(cleaned,df2)    
    



##    tagged_movie_titles = stem_rev_lab['review'].str.split().map(pos_tag)
##    print(tagged_movie_titles.head())
##
main()

main2()
##main2()
###asighment from hereeeeeeeeeeeeeeeeeeeeeeeeeee
    


    
