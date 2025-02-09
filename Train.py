# 1. Zaroori libraries import karo
import numpy as np                                 # Numerical operations ke liye
from sklearn.model_selection import train_test_split  # Data ko train aur test set mein split karne ke liye
from sklearn.linear_model import LogisticRegression     # Logistic Regression model ke liye
from sklearn.metrics import accuracy_score             # Model ki accuracy measure karne ke liye

# 2. Simple dataset create karo
# X: features (ek number hai)
# y: labels (0 ya 1)
X = np.array([[1], [2], [3], [4]])    # 4 samples, har sample ek feature ke saath
y = np.array([0, 0, 1, 1])             # Labels: pehle 2 samples class 0, aakhri 2 samples class 1

# 3. Data ko training aur testing sets mein split karo
# Yahan hum 50% training aur 50% testing use karenge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 4. Logistic Regression model banake, train karo
model = LogisticRegression()        # Model banaya, default settings ke saath
model.fit(X_train, y_train)           # Training data se model ko sikhaya

# 5. Testing set par prediction karo
y_pred = model.predict(X_test)        # Model se testing data ke liye prediction li

# 6. Model ki accuracy check karo
accuracy = accuracy_score(y_test, y_pred)  # Actual labels vs predicted labels compare karo
print("Accuracy:", accuracy)          # Accuracy print karo
