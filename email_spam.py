import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
url = r"C:/Users/ganes/Downloads/SpamBase/spambase.csv"
df = pd.read_csv(url)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print("Dataset Shape:", df.shape)
print("Spam Emails:", sum(y == 1))
print("Ham Emails:", sum(y == 0))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
plt.figure(figsize=(5,5))
y.value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['Ham', 'Spam']
)

plt.title("Ham vs Spam Distribution")
plt.ylabel("")
plt.show()
plt.figure(figsize=(8,5))
feature_means = X.mean().sort_values(ascending=False)[:10]
plt.barh(feature_means.index, feature_means.values)
plt.title("Imp Features")
plt.xlabel("Average Value")
plt.tight_layout()
plt.show()
