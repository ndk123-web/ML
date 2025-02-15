import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

emails = pd.read_csv('messages.csv')

x = emails['message']
y = emails['label']

''' ( TfidfVectorizer )
    - It's Conversion of Text to Numerical list 
    - Ex : 'hello i am ndk' , 'bro how r u' -> 
    - total sub-features in messages ['hello','how','bro',]
    - row look like -> [ [1,0,0] [1,1,0]  ] after vectorize
    
    Before , we transform text to some numerical data 
    1. stop_words  -> stopwords(is,the,a,not) ignored
    2. min_df      -> that will be word atleast 2 times must be in entire dataset that
                      will count as a unique feature 
    3. max_featurs -> if features are 10k then it adds most frequent words as a feature 
'''
vectorizer = TfidfVectorizer(stop_words='english' , min_df=2 , max_features=5000 )
x = vectorizer.fit_transform(x)

'''
    ( RandomOverSampler or SMOTE )
    - The Dataset feature may be dominates if it's not balanced the labels
    - EX : class 0 -> 300 and class 1 -> 1000
    - Here class 1 -> dominates and may model give false spam
    - Using these we duplicates minority class and add into the dataset to balance class
'''
x , y = RandomOverSampler().fit_resample(x.toarray(),y)

model1 = LogisticRegression(max_iter=1000)
model2 = MultinomialNB()

'''
    Train -> 60 %
    Valid -> 20 % 
    Test  -> 20 %
'''
x_train , x_temp , y_train , y_temp = train_test_split(x,y,random_state=42,test_size=0.6)
x_valid , x_test , y_valid , y_test = train_test_split(x_temp,y_temp,random_state=42,test_size=0.2)

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)

y_valid_pred = model1.predict(x_valid)
print("Valid Pred Logistic : " , accuracy_score(y_valid,y_valid_pred))

y_test_pred = model1.predict(x_test)
print("Test Pred  Logistic : ",accuracy_score(y_test,y_test_pred))

y_valid_pred = model2.predict(x_valid)
print("Valid Pred MultiBay : " , accuracy_score(y_valid,y_valid_pred))

y_test_pred = model2.predict(x_test)
print("Test Pred  MultiBay : ",accuracy_score(y_test,y_test_pred))

# Testing Externally and 90+ % is true 
inp = [
    "Congratulations! You have won a free iPhone. Click the link to claim now.",  # Spam (1)
    "Hi John, can we reschedule our meeting for tomorrow morning?",  # Non-Spam (0)
    "You have been selected for a special discount offer! Call now.",  # Spam (1)
    "Dear customer, your bank account needs urgent verification. Login here: fakebank.com",  # Spam (1)
    "Let's meet at the coffee shop at 5 PM. Looking forward to it!",  # Non-Spam (0)
    "Urgent: Your account has been compromised! Reset your password immediately.",  # Spam (1)
    "Hey, I saw your blog post on AI. Great insights!",  # Non-Spam (0)
    "Exclusive deal! Buy 1 get 1 free on all products. Limited time only!",  # Spam (1)
    "Can you please send me the project report by EOD?",  # Non-Spam (0)
    "Reminder: Your doctor's appointment is scheduled for tomorrow at 10 AM.",  # Non-Spam (0)
]
labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
# ans Lo [1  0  1  1  0  0  0  1  0  0]]  1 false
# ans MM [1  0  1  1  1  1  0  1  1  0]   2 false
# extern [1  0  1  1  0  0  0  1  0  0]     

'''
    vectorizer like TfidfVectorizer returns sparse matrix
    - sparse -> low memory usage , avoid zeroes
    - Dense  -> high memory usage , doesn't avoid zeroes
    
    - using toarray() we convert into np.array() object ,
    - because model object is work with dense object
    
'''
inp = vectorizer.transform(inp).toarray()

pred_inp1 = model1.predict(inp)
prob_inp1 = model1.predict_proba(inp)

pred_inp2 = model2.predict(inp)
prob_inp2 = model2.predict_proba(inp)

print("Prediction Logistic : ", pred_inp1)
print("Probability Logistic : ", prob_inp1)

print("Prediction MultiBay : ", pred_inp2)
print("Probability MultiBay : ", prob_inp2)

joblib.dump(model1,"Logistic_Model.pkl")
joblib.dump(model2,"MultiBay_Model.pkl")
joblib.dump(vectorizer,"Tfidvector.pkl")