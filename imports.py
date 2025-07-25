# Step 1: Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# Step 2: Load Data
df = pd.read_csv("customers.csv")  # Replace with your data path

# Step 3: Simple Feature Engineering
features = ['feature1', 'feature2', 'feature3']  # Update as per your dataset
target = 'purchase'  # 1 if purchase, 0 if no purchase

X = df[features]
y = df[target]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
