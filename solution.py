#! /usr/bin/env python3
'''
This file must be executed with python3

For instance `2/3 == 0` in python2 whereas `2/3 > 0` in python3

me@fredefox.eu /^._
 ,___,--~~~~--' /'~
 `~--~\ )___,)/'
     (/\\_  (/\\_
'''
import sqlite3
from numpy import \
    matrix, append, ones, \
    transpose, dot, subtract, \
    outer, argsort
from numpy.linalg import inv, norm, eig
from random import choice
conn = sqlite3.connect("data.sqlite3")

def get_data(sql):
    return conn.execute(sql).fetchall()

# Also available as numpy.mean
def mean(y):
    return sum(y)/len(y)

# Also available as numpy.var
def variance(y):
    m = mean(y)
    return sum(list(map(lambda y: (y - m)**2, y))) / len(y)

def lin_reg_params(S):
    """Returns the parameters of a linear regression model given
    data `S` The returned object is a numpy-matrix.

    Please note that I'm trying to use the names from
    the lecture notes but it's a bit difficult to read because
    the x-vectors are in bold differing them from the y's that are
    just values and are not emphasised """
    y = transpose(matrix(list(map(lambda s: s[1], S))))
    # `x` is a matrix
    x = matrix(list(map(lambda s: s[0], S)))
    z = ones((x.shape[0], 1), int)
    X = append(x, z, axis=1)
    X_t = transpose(X)
    t = inv(dot(X_t, X))
    w = dot(dot(t, X_t), y)
    return w

def affine_model(w):
    w_t = transpose(w)
    n, _ = w.shape
    b = w_t[0, n-1]
    w_t = w_t[0, 0:n-1]
    return lambda x: (dot(w_t, x) + b)[0,0]

def lin_reg(S):
    """This is algorithm 3 from the book"""
    return affine_model(lin_reg_params(S))

def mse(S, f):
    """This function calculates the mean-squared-error
    S is on the form:
    [(x0, y0), ..., [(xn, yn)] \in (X * Y)^n
    f is on the form:
    f : x -> y
    where x \in X, y \in Y
    Y is the set of real numbers"""
    y = transpose(matrix(list(map(lambda s: s[1], S))))
    x = matrix(list(map(lambda s: s[0], S)))
    g = lambda s: (s[1] - f(s[0]))**2
    s = sum(list(map(g, S)))
    return s / len(S)

def euclid(x, y):
    return norm(subtract(x, y))

# This funciton might look more pretty with a fold
# the pythonic name for fold is `reduce`
def argmin(S, f):
    # Right identity for `<`
    m = float("inf")
    res = None
    for s in S:
        s_ = f(s)
        if s_ < m:
            m = s_
            res = s
    return res

def nearest_neighbor(d, S, x):
    (x_min, y_min) = argmin(S, lambda pair: d(pair[0], x))
    return y_min

def pca(data, m):
    S_len = len(data)
    data = list(map(lambda x: matrix(x).T, data))
    mn = sum(data)/S_len
    l = list(map(lambda x: outer(x - mn, x - mn), data))
    S = sum(l)/S_len
    eig_val, eig_vec = eig(S)
    # I saw this in Jespers solution
    # TODO: It *should* be necessary to sort the result from `eig`
    # since it does not guarantee the order of the eigen-vectors
    # [http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html]
    #idx = argsort(eig_val[::-1])
    #eig_val = eig_val[idx]
    #eig_vec = eig_vec[idx]
    # U_m is defined to have the first `m` eigenvectors
    # This is a super-weird syntax for slicing but it should work
    U_m = eig_vec[0:,0:m]
    z = list(map(lambda x: U_m.T.dot(x - mn), data))
    # dec : \mathbb{R}^m -> \mathbb{R}^n
    dec = lambda x: mn + U_m.dot(x)
    # dec : \mathbb{R}^n -> \mathbb{R}^m
    enc = lambda x: U_m.T.dot(x - mn)

    # TODO: Place this somewhere else!
    import matplotlib.pyplot as plt
    # plt.scatter()
    from numpy import cumsum
    cumulative = cumsum(eig_val/sum(eig_val))
    x, y = zip(*list(enumerate(eig_val)))
    plt.title("Eigen spectrum")
    plt.scatter(x, y)
    plt.show()
    from bisect import bisect
    prec = .9
    no = next(i for i, e in enumerate(cumulative) if e >= 0.9)+1
    print("We need {} eigen-vectors to account for {} of the variance". \
        format(no, prec))
    return mn, U_m, z, dec, enc

def cluster(S, k):
    """ S: array of vectors
    k: number of clusters
    returns: [(my, s)] where my is the centroid and s is the cluster"""
    S = list(map(matrix, S))
    prev = list((choice(S), []) for _ in range(k))
    m, _ = prev[0]
    # Assert k>0
    prev[0] = (m, S)
    while True:
        curr = partition(prev)
        curr = reloc_ctr(curr)
        # Now check if the partitions have changed
        compr = list(() for (a, b) in zip(prev, curr) if a[1] != b[1])
        # `compr` is empty iff all partitions are equal
        if not compr:
            break
        prev = curr
    return curr

def partition(S):
    from itertools import chain
    all_points = []
    for _, s in S:
        all_points.extend(s)
    all_means = list(map(lambda pr: pr[0], S))
    res = list((m, []) for m, _ in S)
    for p in all_points:
        i, m = argmin(enumerate(all_means), lambda pr: euclid(pr[1], p))
        res[i][1].append(p)
    return res

def reloc_ctr(S):
    for i, tpl in enumerate(S):
        S_i = tpl[1]
        if not S_i:
            # S_i is empty
            # TODO: What should be the new mean if the set is empty?
            n_m = 0
        else:
            n_m = mean(S_i)
        S[i] = (n_m, S_i)
    return S

if __name__ == "__main__":
    # Question 1
    # ==========
    # Question 1.2
    # ------------
    # sample mean
    # `ys` are a list of vectors
    ys = get_data("SELECT * FROM Redshift_Train_Y")
    # `y_values` is a list of values
    y_values = list(map(lambda x: x[0], ys))
    s_mean = mean(y_values)
    print("Sample mean: {}".format(s_mean))
    # sample variance
    s_var = variance(y_values)
    print("Sample variance: {}".format(s_var))
    # Question 1.3
    # ------------
    # ### Question 1.3.1 ###
    xs = get_data("SELECT * FROM Redshift_Train_X")
    zipd = list(zip(xs, y_values))
    reg_params = lin_reg_params(zipd)
    print("Linear regression parameters:\n{}".format(reg_params))
    # ### Question 1.3.2 ###
    l_reg = lin_reg(zipd)
    print("Mean squared error: {}".format(mse(zipd, l_reg)))
    # ### Question 1.3.3 ###
    # Now we calculate the same quantities just on the *test* data.
    xs = get_data("SELECT * FROM Redshift_Test_X")
    ys = get_data("SELECT * FROM Redshift_Test_Y")
    y_values = list(map(lambda x: x[0], ys))
    zipd = list(zip(xs, y_values))
    # I'm supposing that I shouldn't train another model.
    # l_reg2 = lin_reg(zipd)
    print("Mean squared error (test-data): {}".format(mse(zipd, l_reg)))
    # Question 2
    # ==========
    # Question 2.1
    # ------------
    keystrokes_x = get_data("SELECT * FROM Keystrokes_Train_X;")
    keystrokes_y = get_data("SELECT * FROM Keystrokes_Train_Y;")
    zipd = list(zip(keystrokes_x, keystrokes_y))
    # Map over Test_X
    keystrokes_test_x = get_data("SELECT * FROM Keystrokes_Test_X;")
    l = list(map(lambda x:
        nearest_neighbor(euclid, zipd, x),
        keystrokes_test_x))
    # Compare to Test_Y
    keystrokes_test_y = get_data("SELECT * FROM Keystrokes_Test_Y;")
    zipd = list(zip(l, keystrokes_test_y))
    l = list(map(lambda pair: pair[0] == pair[1], zipd))
    prec = l.count(True)/len(l)
    print("Nearest-neighbor precision: {}".format(prec))
    # Question 2.2
    # ------------
    keystr = get_data("SELECT * FROM Keystrokes_Train_X")
    # Perform PCA
    mn, princ_comp, proj, dec, enc = pca(keystr, 2)
    # TODO: Refactor the plotting of the eigenspectrum.
    # The eigenspectrum is examplified on slide 21 in lecture slide
    # Principal Component Analysis
    # Project all observation down in the plane
    encoded = list(map(lambda x: enc(matrix(x).T), keystr))
    # encoded now is a list [(x, y)] of the all the observation
    # mapped down to \mathbb{R}^2
    import matplotlib.pyplot as plt
    tpls = list(map(lambda x: (x[0,0], x[1,0]), encoded))
    xs, ys = list(zip(*tpls))
    # Plot the things
    plt.title("Keystroke test data")
    plt.scatter(xs, ys)
    plt.show()
    # TODO: Refactor showing how many components are necessary
    # to "explain 90 % of the variance"a
    # The asnwer is on slide 21 in lecture slide Principal Component Analysis
    # Read sec. 5.2 in the lecture notes to understand this
    # Question 2.3
    # ------------
    keystr = get_data("SELECT * FROM Keystrokes_Train_X")
    clstrs = cluster(keystr, 2)
    plt.title("Clustering analysis Keystroke Training Data")
    plt.rc("axes", color_cycle=["c", "m", "y", "k"])
    from itertools import cycle
    # I can run this multiple times and get different results because
    # I choose the initial cluster-centers randomly which aparantly
    # has an impact on the resulting clusters
    # TODO: Do better color-cycling
    for (m, clstr), clr in zip(clstrs, cycle(["c", "m", "y", "k"])):
        clstr_proj = list(map(lambda x: enc(x.T), clstr))
        clstr_proj = list(map(lambda x: (x[0,0], x[1,0]), clstr_proj))
        xs, ys = list(zip(*clstr_proj))
        plt.scatter(xs, ys, color = clr)
    plt.show()
