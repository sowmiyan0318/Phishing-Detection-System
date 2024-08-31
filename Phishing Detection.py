import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('phishing_dataset.csv')

# Check the column names in your DataFrame
print(df.columns)

# Replace 'content' and 'label' with the actual column names
content_column = "url"
label_column = "status"

# Convert the content column to string type
df[content_column] = df[content_column].astype(str)

# Preprocess the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[content_column])
y = df[label_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
