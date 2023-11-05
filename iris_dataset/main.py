import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.frame import DataFrame
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Ustawienie stałego seedu dla powtarzalności wyników
seed = 7
np.random.seed(seed)

# Funkcja do wyświetlania informacji o zbiorze danych
def dataset_info(data: DataFrame) -> None:
    print(data.shape, "\n")
    print(data.describe(), "\n")
    print(data.groupby('class').size(), "\n")
    print(data.groupby('class').mean())

# Funkcja do wyświetlania macierzy wykresów punktowych i histogramów
def scatter_plot_matrix(data: DataFrame) -> None:
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = 9,6
    sns.set(style = 'ticks')
    sns.pairplot(data, hue = 'class', diag_kind='hist')
    plt.show()

# Funkcja do wyświetlania wykresu pudełkowego
def box_whisker_plot(data: DataFrame) -> None:
    plt.rcParams['figure.figsize'] = 9,6
    plt.style.use('fivethirtyeight')
    data.plot(kind='box', sharex=False, sharey=False)
    plt.show()

def plot_roc_curves(model_name, model, X_test, Y_test, class_labels):
    # Binaryzuje dane testowe
    Y_test = label_binarize(Y_test, classes=class_labels)
    n_classes = Y_test.shape[1]

    classifier = OneVsRestClassifier(model)

    classifier.fit(X_test, Y_test)

    # Liczę krzywą ROC i pole ROC dla każdej klasy
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if model_name != "model_SVC1" and model_name != "model_SVC2":
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], classifier.predict_proba(X_test)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], classifier.decision_function(X_test)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])    
        
    # Rysuję krzywe ROC
    plt.figure()
    colors = ['darkorange', 'aqua', 'cornflowerblue']  # Definiuję kolory dla każdej klasy

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Funkcja do przygotowania danych
def prepare_data(data: DataFrame) -> None:
    X = data.values[:,:4]
    Y = data.values[:,4]
    validation_size = 0.30
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    return X_train, X_test, Y_train, Y_test

# Funkcja do wyświetlania wyników walidacji krzyżowej
def model_cross_validation(model, X_train, Y_train, random_state, num_splits=10):
    kfold = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    return cv_results

# Funkcja dostrajania hiperparametrów K-najbliższych sąsiadów
def hyperparameter_tuning_KNN(X_train, Y_train):
    n_neighbors_range = range(5, 16)
    accuracy_scores = []

    for n_neighbors in n_neighbors_range:
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', p=3)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_scores.append(accuracy)

    plt.figure()
    plt.plot(n_neighbors_range, accuracy_scores, marker='o')
    plt.title('KNN Model - Hyperparameter Tuning')
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

# Funkcja dostrajania hiperparametrów drzewa decyzyjnego
def hyperparameter_tuning_DT(X_train, Y_train):
    min_samples_leaf_range = range(1, 5)  
    accuracy_scores = []

    for min_samples_leaf in min_samples_leaf_range:
        model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=seed)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_scores.append(accuracy)

    plt.figure()
    plt.plot(min_samples_leaf_range, accuracy_scores, marker='o')
    plt.title('Decision Tree Model - Hyperparameter Tuning')
    plt.xlabel('min_samples_leaf')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

# Funkcja dostrajania hiperparametrów regresji logistycznej
def hyperparameter_tuning_LR(X_train, Y_train):
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
    accuracy_scores = []

    for solver in solvers:
        model = LogisticRegression(solver=solver, max_iter=1000)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_scores.append(accuracy)

    plt.figure()
    plt.bar(solvers, accuracy_scores)
    plt.title('Logistic Regression Model - Hyperparameter Tuning')
    plt.xlabel('Solver')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.0, 1.01, 0.05))
    plt.grid()
    plt.show()

# Funkcja dostrajania hiperparametrów maszyny wektorów nośnych
def hyperparameter_tuning_SVC(X_train, Y_train):
    gammas = ['auto', 'scale', 0.001, 0.01, 0.1, 1]
    accuracy_scores = []

    for gamma in gammas:
        model = svm.SVC(gamma=gamma, kernel='rbf')
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_scores.append(accuracy)

    plt.figure()
    plt.bar([str(gamma) for gamma in gammas], accuracy_scores)
    plt.title('SVC Model - Hyperparameter Tuning')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.0, 1.01, 0.05))
    plt.grid()
    plt.show()

# Wczytanie zbioru danych
dataset = pd.read_csv('./iris_dataset/data/Iris.csv')
dataset = dataset.drop(columns=['Id'])
feature_names = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'class']
class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
dataset.columns = feature_names
X_train, X_test, Y_train, Y_test = prepare_data(dataset)

# Wyświetlenie informacji o zbiorze danych
# dataset_info(dataset)
# scatter_plot_matrix(dataset)
# box_whisker_plot(dataset)
# hyperparameter_tuning_KNN(X_train, Y_train)
# hyperparameter_tuning_DT(X_train, Y_train)
# hyperparameter_tuning_LR(X_train, Y_train)
# hyperparameter_tuning_SVC(X_train, Y_train)

# Wyświetlenie wyników walidacji krzyżowej
models = [
    ("model_KNN1", KNeighborsClassifier(n_neighbors=13, weights='distance', p=3)),
    ("model_KNN2", KNeighborsClassifier(n_neighbors=6, weights='uniform', p=2)),
    ("model_LR1", LogisticRegression(solver='lbfgs', max_iter=1000)),
    ("model_LR2", LogisticRegression(solver='liblinear', max_iter=1000)),
    ("model_DT1", DecisionTreeClassifier(min_samples_leaf=3)),
    ("model_DT2", DecisionTreeClassifier(min_samples_leaf=5)),
    ("model_SVC1", svm.SVC(gamma='auto', kernel='rbf')),
    ("model_SVC2", svm.SVC(gamma='scale', kernel='rbf'))
]

# results = []
# for name, model in models:
#     cv_results = model_cross_validation(model, X_train, Y_train, random_state=seed)
#     mean_accuracy = cv_results.mean()
#     std_accuracy = cv_results.std()
#     results.append((name, mean_accuracy, std_accuracy))

# pd.set_option('display.colheader_justify', 'center')
# results_df = pd.DataFrame(results, columns=["Model Name", "Mean Accuracy", "Standard Deviation"])
# print(results_df.to_string(index=False))

# Wyświetlenie krzywych ROC
# for name, model in models:
#     plot_roc_curves(name, model, X_test, Y_test, class_labels)