import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
data=pd.read_csv("data.csv",encoding='latin-1')
data=data[['Category','Message']]
data=data.dropna()
data['Category']=data['Category'].map({'ham':0,'spam':1})
data=data[data['Category'].isin([0,1])]
X=data['Message']
y=data['Category']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
vectorizer=TfidfVectorizer(max_features=4000,stop_words='english')
X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)
nb=MultinomialNB()
nb.fit(X_train_tfidf,y_train)
nb_pred=nb.predict(X_test_tfidf)
print("Naive Bayes Accuracy:",accuracy_score(y_test,nb_pred))
print(classification_report(y_test,nb_pred))
lr=LogisticRegression(max_iter=300)
lr.fit(X_train_tfidf,y_train)
lr_pred=lr.predict(X_test_tfidf)
print("Logistic Regression Accuracy:",accuracy_score(y_test,lr_pred))
print(classification_report(y_test,lr_pred))
cm=confusion_matrix(y_test,lr_pred)
plt.imshow(cm)
plt.colorbar()
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j,i,cm[i,j],ha='center',va='center')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
sample=["Congratulations! You have won a free lottery"]
sample_vec=vectorizer.transform(sample)
print("Sample Prediction:",lr.predict(sample_vec)[0])