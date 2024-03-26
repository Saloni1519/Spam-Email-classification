from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data (replace with your own data)
emails = [
    ("This is a promotional email offering a discount on shoes.", "spam"),
    ("Important meeting tomorrow at 10 AM. Please be prepared to discuss the project.", "ham"),
    ("Win a free iPhone! Click here to claim your prize.", "spam"),
    ("Hi John, how are you doing? Just checking in.", "ham"),
    ("URGENT! Your account has been suspended. Please click here to verify your identity.", "spam"),
]

# Separate features (text) and target variable (spam/ham)
X = [email[0] for email in emails]
y = [email[1] for email in emails]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_features, y_train)

# Test model accuracy
predictions = model.predict(X_test_features)
accuracy = model.score(X_test_features, y_test)

print("Model Accuracy:", accuracy)

# Classify a new email (replace with your own email text)
new_email = "This is a promotional email offering a discount on travel."
new_email_features = vectorizer.transform([new_email])
prediction = model.predict(new_email_features)

if prediction[0] == 'spam':
  print("This email is classified as spam!")
else:
  print("This email is classified as ham (not spam).")
