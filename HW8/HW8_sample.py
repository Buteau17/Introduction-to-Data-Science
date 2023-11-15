# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# lets create a text
text = "The fries were great too."

# length of text ( includes spaces)
print("length of text: ",len(text))

# split the text
splitted_text = text.split() # default split methods splits text according to spaces
print("Splitted text: ",splitted_text)    # splitted_text is a list that includes words of text sentence
# each word is called token in text maning world.

# find specific words with list comprehension method
specific_words = [word for word in splitted_text if(len(word)>2)]
print("Words which are more than 3 letter: ",specific_words)

# capitalized words with istitle() method that finds capitalized words
capital_words = [ word for word in splitted_text if word.istitle()]
print("Capitalized words: ",capital_words)

# words which end with "o": endswith() method finds last letter of word
words_end_with_o =  [word for word in splitted_text if word.endswith("o")]
print("words end with o: ",words_end_with_o) 

# words which starts with "w": startswith() method
words_start_with_w = [word for word in splitted_text if word.startswith("w")]
print("words start with w: ",words_start_with_w) 

# unique with set() method
print("unique words: ",set(splitted_text))  # actually the word "no" is occured twice bc one word is "no" and others "No" there is a capital letter at first letter

# make all letters lowercase with lower() method
lowercase_text = [word.lower() for word in splitted_text]

# then find uniques again with set() method
print("unique words: ",set(lowercase_text))

# chech words includes or not includes particular substring or letter
print("Is f letter in fries word:", "f" in "fries")

# check words are upper case or lower case
print("Is word uppercase:", "FRIES".isupper())
print("Is word lowercase:", "great".islower())

# check words are made of by digits or not
print("Is word made of by digits: ","12345".isdigit())

# get rid of from white space characters like spaces and tabs or from unwanted letters with strip() method
print("00000000were great: ","00000000were great".strip("0"))

# find particular letter from front 
print("Find particular letter from back: ","Were great too".find("o"))  # at index 1

# find particular letter from back  rfind = reverse find
print("Find particular letter from back: ","Were great too".rfind("o"))  # at index 8

# replace letter with letter
print("Replace o with 3 ", "Were great too".replace("o","3"))

# find each letter and store them in list
print("Each letter: ",list("Were great"))

# Cleaning text
text1 = "    Not tasty and the texture was just nasty    "
print("Split text: ",text1.split(" "))   # as you can see there are unnecessary white space in list

# get rid of from these unnecassary white spaces with strip() method then split
print("Cleaned text: ",text1.strip().split(" "))

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.head()

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

callouts = [word for word in text.split(" ") if re.search("[^a-zA-Z]",word)]
print("callouts: ",callouts) 
callouts1 = [word for word in text.split(" ") if re.search("@\w+",word)]
print("callouts: ",callouts1)

# find specific characters like "w"
print(re.findall(r"[w]",text))
# "w"ith, "w"indo"w", sho"w"ing, s"w"itches 

# do not find specific character like "w". We will use "^" symbol
print(re.findall(r"[^w]",text))

print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)