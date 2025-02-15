import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

model_logistic = joblib.load('Logistic_Model.pkl')
vectorizer = joblib.load('Tfidvector.pkl')

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
# extern [1  0  1  1  0  0  0  1  0  0]     # loading and working fine

inp = vectorizer.transform(inp).toarray()

y_pred = model_logistic.predict(inp)
print(y_pred)