import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset from CSV
data = pd.read_csv("digits_dataset.csv")

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

# Visualize some sample digits
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for ax, (index, row) in zip(axes, data.iterrows()):
    img = row[:-1].values.reshape(8, 8)  # Reshape to 8x8 image
    ax.set_axis_off()
    ax.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {row['label']}")
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Test the model with a custom input
sample = X.iloc[5].values.reshape(8, 8)  # Original image
plt.imshow(sample, cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Original Label: {y.iloc[5]}")
plt.show()

# Predict the label
sample_input = scaler.transform([X.iloc[5]])
sample_prediction = model.predict(sample_input)
print(f"Predicted Label: {sample_prediction[0]}")
