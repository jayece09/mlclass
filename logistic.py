"""
Logistic regression classifier in python.
"""
import operator
import math
import random

def inner_prod(v1, v2):
     'inner production of two vectors.'
     sum = 0
     for i in xrange(len(v1)):
            sum += v1[i] * v2[i]
     return sum

def matmult4(m, v):
     'matrix multiply vector by inner production.'
     return [inner_prod(r, v) for r in m]

def vecadd(v1, v2): return map(operator.add, v1, v2)

def vecsub(v1, v2): return map(operator.sub, v1, v2)

def transpose(m): return map(list, zip(*m))

def sigmoid(z): return (1 + math.exp(-z)) ** (-1)

def predict(theta, x):
    p = (sigmoid(sum(map(operator.mul, theta, x))))

    return "+1" if p > 0.5 else "-1"

def hypothesis(theta, example): return sigmoid(inner_prod(theta, example))

def grad_descent(X, y, theta, alpha, lambda_ = 1.0,  iters=5):
    """
    Batch gradient descent.
    """
    n = len(theta)
    m = len(X)
    for itrn in range(iters):
        h = map(sigmoid, matmult4(X, theta))
        for j in range(n):
            theta[j] -= (alpha / m) * matmult4([[q] for q in vecsub(h, y)], [row[j] for row in X])[0]
            theta[j] -= (alpha*lambda_/m) * theta[j]
    return theta

def train(X, y):
    X = normalize(X)
    return grad_descent(X, y, [0 for i in range(len(X[1]))], .1, 100,  100)

mean = lambda xs: 1.0 * sum(xs)/len(xs)

def stdev(xs):
    mu = mean(xs)
    return math.sqrt(sum((x-mu)**2 for x in xs) / len(xs))

def normalize(X):
    for j, col in enumerate(transpose(X[:])):
        mu, sigma = mean(col), stdev(col)
        if not sigma: sigma += 1
        for i, item in enumerate(col):
            X[i][j] = (X[i][j] - mu) / sigma
    return [[1]+row for row in X]

def randomize(X, y):
    m = range(len(X))
    random.shuffle(m)
    return [X[i] for i in m], [y[i] for i in m]

def cost_func(X, y, theta):
    m = len(X)
    h = map(sigmoid, matmult4(X, theta))
    J = -(1.0/m) * ( matmult4([[q] for q in y], map(math.log, h))[0] +
                     matmult4([[1-item] for item in y], [math.log(1 - item) for item in h])[0])
    return J


"""
Example usage:
Note: Normalize both training and test data before learning and classification.
"""
def main():
    num_cases, _  = map(int, raw_input().split())
    X, y = [], []
    labels, test = [], []
    for inp in range(num_cases):
        _, cls, features = raw_input().strip().split(" ", 2)
        y.append(1 if cls =="+1" else 0)
        X.append([float(item.split(':')[1]) for item in features.split()])
    test_num = int(raw_input())
    for i in range(test_num):
        label, features = raw_input().strip().split(" ", 1)
        labels.append(label)
        test.append([float(item.split(':')[1]) for item in features.split()])
    test = normalize(test)
    theta = train(X, y)
    for label, case in zip(labels, test):
        print label, predict(theta, case)

if __name__ == '__main__':
    main()
