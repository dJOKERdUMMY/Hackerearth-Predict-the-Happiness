import pandas as pd
#reading the training file
train = pd.read_csv("train.csv")

#exploratory data analysis for knowing the type of data
#uncomment these lines of code to see the actual output for data analysis
#train.head()
#train.Browser_Used.value_counts()
#found error in data collection as most browsers with same name has been used again with different ID . we need to merge it.
#print("data-type :",type(train.Browser_Used.value_counts()))
#browsers = train.Browser_Used.value_counts().index
#print("data-type of browsers :",type(browsers))
#print("Name of Browsers :")
#for i in browsers:
#    print(i)

#cleaning the data because there is multiple use of same browser with different names
train["Browser_Used"] = train["Browser_Used"].str.replace('Mozilla Firefox', 'Firefox')
train["Browser_Used"] = train["Browser_Used"].str.replace('Mozilla','Firefox')
train["Browser_Used"] = train["Browser_Used"].str.replace('Internet Explorer', 'InternetExplorer')
train["Browser_Used"] = train["Browser_Used"].str.replace('IE', 'InternetExplorer')
train["Browser_Used"] = train["Browser_Used"].str.replace('Google Chrome', 'Chrome')
print("browsers managed...........")

#train.Browser_Used.value_counts(dropna=False)
#train.Is_Response.value_counts(dropna=False)
#train.groupby(['Browser_Used','Is_Response'])['Is_Response'].count()
#train.groupby(['Device_Used','Is_Response'])['Is_Response'].count()
#train.groupby(['Browser_Used','Device_Used','Is_Response'])['Is_Response'].count()
#using only the comments to train the model

df = train.drop(['Device_Used','Browser_Used'],axis=1)
print(df.head(5))

#importing nltk for NLP usage
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

#removing stopwords
def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review,'html.parser').get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

clean_review = review_to_words(train["Description"][0])
#testing if the function defined works properly
print(clean_review)

num_reviews = df["Description"].size
clean_train_reviews = []
for i in range( 0, num_reviews ):
    clean_train_reviews.append(review_to_words(df["Description"][i]))

print("Creating bag of words model")
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = 'word',
                            tokenizer = None,
                            preprocessor = None,
                            stop_words = None,
                            max_features = 1000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print(train_data_features.shape)

vocab = vectorizer.get_feature_names()
print(vocab[0:20])

import numpy as np
dist = np.sum(train_data_features,axis=0)
for tag,count in zip(vocab,dist):
    print(count,tag)

#creating ML classifier to fit and predict the data
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features,df['Is_Response'])

test = pd.read_csv('./data/test.csv')
#print(test.head())

df = test.drop(['Browser_Used','Device_Used'],axis=1)
#print(df.head())

num_reviews = len(test.Description)
clean_test_reviews = []

for i in range(0,num_reviews):
    clean_review = review_to_words( test["Description"][i] )
    clean_test_reviews.append(clean_review)
    
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#predicting the data for output
result = forest.predict(test_data_features)

output1 = test.drop(['Description','Browser_Used','Device_Used'],axis=1)
#print(output1.head())

output1['Is_Response'] = pd.Series(result)
#print(output1.head())

#writing output to file
output1.to_csv('./output/submit.csv',index=False,quoting=3)