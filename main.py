# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:10:49 2017
@author: Lenovo
"""

import pandas as pd
import re, string
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from sklearn import preprocessing as spp

def repIt(x):
    x = x.strip()
    if all(c.isdigit() for c in x):
        return 'number'
    elif any(c.isdigit() for c in x):
        return 'id'
    else:
        return x
        


def my_pp(raw_text_df):
    #Basic Preprocess
    raw_text=raw_text_df.copy()
    processed = raw_text.str.lower()
    #With reg ex
    processed = processed.str.replace(r'.',' ')
    processed = processed.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',' email ')
    processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',' url ')
    processed = processed.str.replace(r'£|\$', ' currency ')   
    processed = processed.str.replace(r' rs\. ', ' currency ')
    processed = processed.str.replace(r' inr ', ' currency ')
    processed = processed.str.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',' phone number ')    
    processed = processed.str.replace(r'(\d{4}[-.]+\d{2}[-.]+\d{2})', ' date ')
    processed = processed.str.replace(r'(\d{2}[:]+\d{2})|(\d{1}[:]+\d{2})', ' time ')
    #With direct matches
    processed = processed.str.replace(r' flt ', ' flight ')
    processed = processed.str.replace(r' n\'t ', ' not ')
    processed = processed.str.replace(r' no\. ', ' number ')
    for row in range(len(processed)):
        temp = str(processed.iloc[row]) 
        temp = re.sub("'s", "", temp)
        temp = re.sub("-", "", temp);temp = re.sub("__", "", temp);temp = re.sub("_", "", temp);
        processed.iloc[row] = ' '.join([repIt(x) for x in wordpunct_tokenize(processed.iloc[row])])
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
    processed = processed.str.replace(r'\s+', ' ')
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    
    #stopwords removal
    stop_words = set(stopwords.words('english'))
    processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in set(stop_words)))

    #Stemmer
    #porter = nltk.PorterStemmer()
    #processed = processed.apply(lambda x: ' '.join(porter.stem(term) for term in x.split()))
    
    #Lemmatize
    wordnet = WordNetLemmatizer()
    for row in range(len(processed)):
        print(row)
        temp = str(processed.iloc[row]) 
        tokens = word_tokenize(temp)
        pos_tags = nltk.pos_tag(tokens)
        index = 0
        for word, tag in pos_tags:
            if tag.startswith("NN"):
                word = wordnet.lemmatize(word,pos='n')
            elif tag.startswith("VB"):
                word = wordnet.lemmatize(word,pos='v')
            elif tag.startswith("JJ"):
                word = wordnet.lemmatize(word,pos='a')
            elif tag.startswith("R"):
                word = wordnet.lemmatize(word,pos='r')
            tokens[index] = word
            index+=1
        temp = ' '.join(tokens)
        processed.iloc[row] = temp
    return processed

#Training and my Validation Data
df = pd.read_csv('(A)TRAIN_SMS.csv')
le = spp.LabelEncoder()
y = df.iloc[:,0]
model = le.fit(y) 
y_enc = model.transform(y)
raw_text_df = df.loc[:,'Message']
#processed = my_pp(raw_text_df)
#all_tokens = word_tokenize( ' '.join(np.array(processed)) )
#nUnique = np.unique(all_tokens)
#Save pickle
#import pickle
#with open('saveThis.pickle','w') as ofile:
#    pickle.dump(processed,ofile)
#Load the pickle
import pickle
with open('saveThis.pickle','r') as ifile:
    processed=pickle.load(ifile)   
#Copute desc vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_ngrams = vectorizer.fit_transform(processed)
#Do test-train division and pca fitting
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ngrams,y_enc,test_size=0.1,random_state=42,stratify=y_enc)
from sklearn.decomposition import TruncatedSVD #clf = PCA(n_components=1000) #Not for csr matrix input
clf = TruncatedSVD(n_components=1000)
clf.fit(X_train)
new_X_train = clf.transform(X_train)
new_X_test = clf.transform(X_test)
#Predict on dev data
dev_data = pd.read_csv('(B)DEV_SMS.csv')
dev_processed  = my_pp(dev_data.loc[:,'Message'])
dev_X_ngrams = vectorizer.transform(dev_processed)
dev_test = clf.transform(dev_X_ngrams)

#99.2%
#Model: SVM
from sklearn import svm
svm_model = svm.LinearSVC(loss='hinge', multi_class='crammer_singer')
svm_model.fit(new_X_train, y_train)
results = svm_model.predict(new_X_test)
from sklearn.metrics import accuracy_score #f1_scr = metrics.f1_score(y_test, results)
f1_scr = accuracy_score(y_test, results)
upload_results = svm_model.predict(dev_test)
upload_results_words = model.inverse_transform(upload_results)

#99.36%
#Model: Logistic Regression
from sklearn.linear_model import LogisticRegressionCV
lrCV_model = LogisticRegressionCV(cv=5,multi_class='multinomial')
lrCV_model.fit(new_X_train, y_train)
results = lrCV_model.predict(new_X_test)
from sklearn.metrics import accuracy_score #f1_scr = metrics.f1_score(y_test, results)
f1_scr = accuracy_score(y_test, results)
upload_results = lrCV_model.predict(dev_test)
upload_results_words = model.inverse_transform(upload_results)

#MLP
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(learning_rate='adaptive',max_iter=400,verbose=True,tol=1e-4)
mlp_model.fit(new_X_train, y_train)
results = mlp_model.predict(new_X_test)
from sklearn.metrics import accuracy_score #f1_scr = metrics.f1_score(y_test, results)
f1_scr = accuracy_score(y_test, results)
upload_results = mlp_model.predict(dev_test)
upload_results_words = model.inverse_transform(upload_results)

#Save as csv file and upload
answerFinal=pd.DataFrame(columns=['RecordNo','Label'])
answerFinal.loc[:,'Label']=pd.Series(upload_results_words)
answerFinal.loc[:,'RecordNo'] = pd.Series(dev_data.loc[:,'RecordNo'])
answerFinal.to_csv('hereItIs_1.csv',header=True, index=False)









#87%
#Model: GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(new_X_train, y_train)
results = gnb_model.predict(new_X_test)
from sklearn.metrics import accuracy_score #f1_scr = metrics.f1_score(y_test, results)
f1_scr = accuracy_score(y_test, results)
upload_results = gnb_model.predict(dev_test)
upload_results_words = model.inverse_transform(upload_results)

#84%
#Model: BernoulliNB
from sklearn.naive_bayes import BernoulliNB
bnb_model = BernoulliNB()
bnb_model.fit(new_X_train, y_train)
results = bnb_model.predict(new_X_test)
from sklearn.metrics import accuracy_score #f1_scr = metrics.f1_score(y_test, results)
f1_scr = accuracy_score(y_test, results)
upload_results = bnb_model.predict(dev_test)
upload_results_words = model.inverse_transform(upload_results)




