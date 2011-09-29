# -*- coding: utf8 -*-
"""
This is a python based implementation of finding the optimum line to fit a
regression problem using the method of gradient descent.
"""
from functools import partial

def derivative(polynomial):
    """
    Returns the derivative of a polynomial in the form a + bx + cx² + dx³ + ... represented as (a, b, c, d)
    """
    return sum(i*x for i, x in enumerate(polynomial))

def value(polynomial, x):
    """
    Returns the numerical value of the polynomial for a particular value of x.
    """
    return sum(a*(x**i) for i, a in enumerate(polynomial))

def correction_term0(alpha, training_set, theta0, theta1):
    return sum(alpha * (theta1*x + theta0 - y) for x, y in training_set)

def correction_term1(alpha, training_set, theta0, theta1):
    return sum(alpha * x * (theta1*x + theta0 - y) for x, y in training_set)

def gradient_descent(alpha, guesses, correction_terms, training_set):
    """
    Generalized algorithm for gradient descent, where :
        `alpha` is the learning rate of the descent

        `guesses` represents a tuple containing the values of thetas to start the gradient descent from,

        `correction_terms` is a tuple of functions that compute the numerical values of the partial derivatives for
         each of the thetas

    Returns the set of local optimum thetas depending on the guesses
    """
    apply_values = lambda f: partial(f, alpha, training_set)
    correction_terms = map(apply_values, correction_terms)
    temp_thetas = guesses
    thetas = guesses
    prev_result = ()
    m = len(training_set)
    while all(abs(x-y) > 0.000001 for x, y in zip(thetas, prev_result)):
        thetas = [x - y/2/m for x, y in zip(thetas, [term(*temp_thetas) for term in correction_terms])]
        prev_result, temp_thetas = temp_thetas, tuple(thetas)
        print thetas, prev_result
    return tuple(thetas)

def linear_regression(training_set, alpha = 0.01):
    """
    Returns a hypothesis polynomial for a given training set
    """
    guesses = (0, 0)
    correction_terms = (correction_term0, correction_term1)
    return gradient_descent(alpha, guesses, correction_terms, training_set)

