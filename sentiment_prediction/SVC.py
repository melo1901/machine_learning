import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Załaduj dataset z pliku CSV
df = pd.read_csv('IMDB_Dataset.csv')

# Zmniejsz dataset do 20% oryginalnej wielkości
df = df.sample(frac=0.2, random_state=42)

# Podziel dataset na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.1, random_state=42)

# Zwektoryzuj recenzje używając reprezentacji TF-IDF(zamieniany słowa z recenzji na liczby, które komputer zrozumie)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Defniuj model (SVC)
model = SVC(probability=True, random_state=0)

# Trenuj model
model.fit(X_train_vec, y_train)

# Przewiduj charakter recenzji na zbiorze testowym
y_pred = model.predict(X_test_vec)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
y_true_binary = y_test.apply(lambda x: 1 if x == 'positive' else 0)

# Upewnij się, że y_pred jest również numeryczny (konwertuj jeśli jest to konieczne)
y_pred_binary = np.where(y_pred == 'positive', 1, 0)

# Ocena modelu z etykietami binarnymi
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
roc_auc = roc_auc_score(y_true_binary, model.predict_proba(X_test_vec)[:, 1])

# Printuj wyniki
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Generuj krzywą ROC
fpr, tpr, thresholds = roc_curve(y_true_binary, model.predict_proba(X_test_vec)[:, 1])

end_time = time.time()
# Rysuj krzywą ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

print(f"Total time taken: {end_time - start_time:.2f} seconds")

#joblib.dump((model, vectorizer), 'model_and_vectorizerKNN.joblib')