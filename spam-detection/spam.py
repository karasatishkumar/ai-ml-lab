import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Read Data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
print(data.head())

# 2. Take class and message from the data
data = data[["class", "message"]]
print(data.head())

# 3. Separate test and training data sets and use Multinomial Naive Bayes algorithm
x = np.array(data["message"])
y = np.array(data["class"])
cv = CountVectorizer()
X = cv.fit_transform(x)
# Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))

# 4. Test the model
sample = 'You won $40 cash price'
data = cv.transform([sample]).toarray()
print(sample)
print(clf.predict(data))
