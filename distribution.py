# Step 1: Identify Class Distribution
print(df[target].value_counts())

# Step 2: Simulate micro-numerosity by creating a small dataset
df_few = resample(df, replace=False, n_samples=30, random_state=42)
X_few = df_few[features]
y_few = df_few[target]

# Step 3: Train Model on Small Dataset
X_train_few, X_test_few, y_train_few, y_test_few = train_test_split(X_few, y_few, test_size=0.2, random_state=42)
clf_few = RandomForestClassifier(random_state=42)
clf_few.fit(X_train_few, y_train_few)
y_pred_few = clf_few.predict(X_test_few)

# Step 4: Evaluate Model with Small Sample
print(classification_report(y_test_few, y_pred_few))
print("Accuracy with micro-numerosity:", accuracy_score(y_test_few, y_pred_few))
