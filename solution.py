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

# This is algorithm 3 from the book
def lin_reg(S):
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
    n, _ = w.shape
    b = w[n-1, 0]
    w = w[0:n-1, 0]
    w_t = transpose(w)
    return lambda x: dot(w_t, x) + b

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
    l_reg = lin_reg(zipd)
    print("Linear regression: {}".format(l_reg))
