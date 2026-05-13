import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
X = iris.data
y = iris.target
flowers = ['Lotus','Hibiscus','Marigold']
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [flowers[i] for i in y]
print(df.head())
sns.pairplot(df, hue='species')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
sample = [[5.1,3.5,1.4,0.2]]
pred = model.predict(sample)
print("Predicted Flower:", flowers[pred[0]])
