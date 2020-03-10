
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt

# class RandomSplitter:
#
#     def split(self, x, y):
#         ### Radnom split
#         all_indeces = np.arange(len(x))
#         left_indeces = np.sort(np.random.choice(all_indeces, np.random.randint(0, len(all_indeces)), replace=False))
#         right_indeces = np.array(list(set(all_indeces) - set(left_indeces)))
#         return np.take(x, left_indeces), np.take(y, left_indeces), np.take(x, right_indeces), np.take(y, right_indeces), lambda x: x > 0

def show_solution(mdl, x, step=1.0):
    x0 = x[:, 0]
    x0_range = np.round(x0.max() - x0.min())
    x1 = x[:, 1]
    x1_range = np.round(x1.max() - x1.min())

    x = np.arange(x0.min(), x0_range + x0.min(), step)
    y = np.arange(x1.min(), x1_range + x1.min(), step)
    x, y = np.meshgrid(x, y)
    xy = np.array([x, y]).transpose((1, 2, 0)).reshape(-1, 2)

    pred = mdl.predict(xy)
    plt.scatter(xy[:, 0], xy[:, 1], c=pred)
    plt.show()


class SVMSplitter:

    def fit_split(self, x, y):
        svm_clf = SVC()
        svm_clf.fit(x, y)
        return svm_clf.predict(x)

class GinniScore:

    def score(self, predicted, actual):
        return 1 - ((~(predicted == actual)).sum() / predicted.shape[0])


class Node:

    def __init__(self, x, y, splitter='default'):
        self.x = x
        self.y = y
        self.splitter = SVMSplitter()
        self.score_calculus = GinniScore()
        self.is_leaf = True

    def make_split(self):
        predicted_array = self.splitter.fit_split(self.x, self.y)
        self.calculated_score = self.score_calculus.score(predicted_array, self.y)

        self.is_leaf = False
        self.x = None
        self.y = None


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data['data']
Y = data['target']

X = X[:, 0:2]

node = Node(X, Y)
node.make_split()




