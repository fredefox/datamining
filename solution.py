'''
me@fredefox.eu /^._
 ,___,--~~~~--' /'~
 `~--~\ )___,)/'
     (/\\_  (/\\_
'''
import sqlite3
from numpy import * #matrix, append, zeros, transpose
from numpy.linalg import inv
conn = sqlite3.connect("data.sqlite3")

# Question 1
def redshift_x():
    return conn.execute("SELECT * FROM Redshift_Train_X").fetchall()

def redshift_y():
    return conn.execute("SELECT * FROM Redshift_Train_Y").fetchall()

def mean(y):
    return sum(y)/len(y)

def variance(y):
    m = mean(y)
    return sum(list(map(lambda y: (y - m)**2, y))) / len(y)

# Returns the parameters of a linear regression model given data `S`
# The returned object is a numpy-matrix
def lin_reg_params(S):
    """ Please note that I'm trying to use the names from
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
    w = w_t[0, 0:n-1]
    return lambda x: dot(w_t, x) + b

# This is algorithm 3 from the book
def lin_reg(S):
    return affine_model(lin_reg_params(S))


if __name__ == "__main__":
    # Question 1
    # sample mean
    # `ys` are a list of vectors
    ys = redshift_y()
    # `y_values` is a list of values
    y_values = list(map(lambda x: x[0], ys))
    s_mean = mean(y_values)
    print("Sample mean: {}".format(s_mean))
    # sample variance
    s_var = variance(y_values)
    print("Sample variance: {}".format(s_var))
    xs = redshift_x()
    zipd = list(zip(xs, y_values))
    reg_params = lin_reg_params(zipd)
    print("Linear regression parameters:\n{}".format(reg_params))
