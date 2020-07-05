import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
import re
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

data = pd.read_csv('text_emotion.csv')
data['sentiment'].value_counts()

# Making all letters lowercase
data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Removing Punctuation, Symbols
data['content'] = data['content'].str.replace('[^\w\s]',' ')

# Removing Stop Words using NLTK
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Lemmatisation
data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#Correcting Letter Repetitions
def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

# Code to find the top 10,000 rarest words appearing in the data
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

# Removing all those rarely appearing words from the data
freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#Encoding output labels
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)

# Splitting into training and testing data in 90:10 ratio
X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1)

# Extracting Count Vectors Parameters
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])

# Dumping the Count Vectorizer in the form of pickel so that we don't have to generate it over & over again
pickle.dump(count_vect, open("vector.pickel", "wb"))

vectorizer = pickle.load(open("vector.pickel", "rb"))
X_train_count =  vectorizer.transform(X_train)
X_val_count =  vectorizer.transform(X_val)

# Model: Linear SVM
lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=30, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('LSVM using count vectors accuracy: %s' % accuracy_score(y_pred, y_val))
# lsvm using count vectors accuracy 0.7928709055876686

# Dumping the model in the form of pickel so that we don't have to train it over & over again
pickle.dump(lsvm, open('model.pickel', 'wb'))

#Below are 8 random statements. The first 4 depict happiness. The last 4 depict sadness
tweets = pd.DataFrame(['I am very happy today! The atmosphere looks cheerful',
                       'Things are looking great. It was such a good day',
                       'Success is right around the corner. Lets celebrate this victory',
                       'Everything is more beautiful when you experience them with a smile!',
                       'Now this is my worst, okay? But I am gonna get better.',
                       'I am tired, boss. Tired of being on the road, lonely as a sparrow in the rain. I am tired of all the pain I feel',
                       'This is quite depressing. I am filled with sorrow',
                       'His death broke my heart. It was a sad day'])

# Doing some preprocessing on these tweets as done before
tweets[0] = tweets[0].str.replace('[^\w\s]',' ')

#from nltk.corpus import stopwords
stop = stopwords.words('english')
tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#from textblob import Word
tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Extracting Count Vectors Parameters
vectorizer = pickle.load(open("vector.pickel", "rb"))
tweet_count = vectorizer.transform(tweets[0])

#Predicting the emotion of the tweet using our already trained linear SVM
model = pickle.load(open('model.pickel', 'rb'))
result = model.predict(tweet_count)
print(result)