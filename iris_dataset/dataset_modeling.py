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
from sklearn import model_selection


class Dataset_Handler():

    def __init__(self, data: DataFrame) -> None:
        self.data = data
        self.feature_names = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'class']
        self.class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self.data.columns = self.feature_names
    
    def dataset_info(self) -> None:
        print(self.data.shape, "\n")
        print(self.data.describe(), "\n")
        print(self.data.groupby('class').size(), "\n")
        print(self.data.groupby('class').mean())

    def scatter_plot_matrix(self) -> None:
        plt.style.use('fivethirtyeight')
        plt.rcParams['figure.figsize'] = 9,6
        sns.set(style = 'ticks')
        sns.pairplot(self.data, hue = 'class')
        plt.show()

    def box_whisker_plot(self) -> None:
        plt.rcParams['figure.figsize'] = 9,6
        plt.style.use('fivethirtyeight')
        self.data.plot(kind='box', sharex=False, sharey=False)
        plt.show()

    def prepare_data(self) -> None:
        X = self.data.values[:,:4]
        Y = self.data.values[:,4]
        validation_size = 0.30
        seed = self.random_state()
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        return X_train, X_test, Y_train, Y_test
    
    def random_state(self) -> int:
        seed = 7
        np.random.seed(seed)
        return seed

    def model_cross_validation(self):
        model_KNN1 = KNeighborsClassifier(n_neighbors=15, weights='distance', p=3)
        model_KNN2 = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
        model_LR1 = LogisticRegression(solver='lbfgs', max_iter=1000)
        model_LR2 = LogisticRegression(solver='liblinear', max_iter=1000)
        model_DT1 = DecisionTreeClassifier(min_samples_leaf=3, random_state=self.random_state())
        model_DT2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=self.random_state())
        model_SVC1 = svm.SVC(gamma='auto', kernel='rbf')
        model_SVC2 = svm.SVC(gamma='scale', kernel='linear')

        models = ['KNN1', 'KNN2', 'LR1', 'LR2', 'DT1', 'DT2', 'SVC1', 'SVC2']
        num_models = len(models)

        scoring = 'accuracy'
        n_splits = 10
        kfold = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state())
        X_train, X_test, Y_train, Y_test = self.prepare_data()
        cv_results = np.zeros((num_models, n_splits))

        for i in range(num_models):
            model = eval('model_' + models[i])
            cv_results[i, :] = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            
        print('\n Accuracy: Mean (std) for Different Models and Parameters: \n')
        for i in range(num_models):
            print_results = "%s: %.4f (%.4f)" % (models[i], cv_results[i, :].mean(), cv_results[i, :].std())
            print(print_results)