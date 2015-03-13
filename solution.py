'''
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
    idx = argsort(eig_val[::-1])
    eig_val = eig_val[idx]
    eig_vec = eig_vec[idx]
    # U_m is defined to have the first `m` eigenvectors
    # This is a super-weird syntax for slicing but it should work
    U_m = eig_vec[0:,0:m]
    z = list(map(lambda x: U_m.T.dot(x - mn), data))
    # dec : \mathbb{R}^m -> \mathbb{R}^n
    dec = lambda x: mn + U_m.dot(x)
    # dec : \mathbb{R}^n -> \mathbb{R}^m
    enc = lambda x: U_m.T.dot(x - mn)
    return mn, U_m, z, dec, enc

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
    keystr = get_data("SELECT * FROM Keystrokes_Test_X")
    # Perform PCA
    mn, princ_comp, proj, dec, enc = pca(keystr, 2)
    # TODO: Plot eigenspectrum
    # Project all observation down in the plane
    encoded = list(map(lambda x: enc(matrix(x).T), keystr))
    # encoded now is a list [(x, y)] of the all the observation
    # mapped down to \mathbb{R}^2
    from matplotlib.pyplot import scatter, show
    tpls = list(map(lambda x: (x[0,0], x[1,0]), encoded))
    xs, ys = list(zip(*tpls))
    # Plot the things
    scatter(xs, ys)
    show()
    # TODO: How many components are necessary to "explain 90 % of the variance"
    # Read sec. 5.2 in the lecture notes to understand this
