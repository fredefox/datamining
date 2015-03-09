'''
me@fredefox.eu /^._
 ,___,--~~~~--' /'~
 `~--~\ )___,)/'
     (/\\_  (/\\_
'''
import sqlite3
from numpy import matrix, append, ones, transpose, dot
from numpy.linalg import inv, norm
conn = sqlite3.connect("data.sqlite3")

def get_data(sql):
    return conn.execute(sql).fetchall()

def mean(y):
    return sum(y)/len(y)

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
