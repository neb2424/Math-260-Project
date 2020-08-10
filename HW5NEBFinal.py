#Q1
import numpy as np
import matplotlib.pyplot as plt
import math

def next(r,y0,t0,t1):
    assert(t1>t0)
    dt = t1-t0
    dy = dt * (r*y0*(1-y0))
    y1 = y0 + dy
    return y1

def rk4 ( dydt, tspan, y0, n ):
  if ( np.ndim ( y0 ) == 0 ):
    m = 1
  else:
    m = len ( y0 )

  tfirst = tspan[0]
  tlast = tspan[1]
  dt = ( tlast - tfirst ) / n
  t = np.zeros ( n + 1 )
  y = np.zeros ( [ n + 1, m ] )
  t[0] = tspan[0]
  y[0,:] = y0

  for i in range ( 0, n ):

    f1 = dydt ( t[i],            y[i,:] )
    f2 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f1 / 2.0 )
    f3 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f2 / 2.0 )
    f4 = dydt ( t[i] + dt,       y[i,:] + dt * f3 )

    t[i+1] = t[i] + dt
    y[i+1,:] = y[i,:] + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

  return t, y

def func_exact(t):
    return 1/(1+math.exp(-t))

def func(t,y):
    return y*(1-y)

def E(h):
    n = int(5/h)
    t,y = rk4(func, [0,5], .5, n)
    y_exact = []
    for t_ in t:
        y_exact.append(func_exact(t_))
    y = np.array(y).reshape((-1,1))
    y_exact = np.array(y_exact).reshape((-1,1))
    error = np.max(np.abs(np.array(y) - np.array(y_exact)))
    input()
    return error


t,y = rk4(func, [0,5], .5, 100)
plt.plot(t,y,linestyle='-', linewidth=1)
plt.xlabel('t')
plt.ylabel('y')
y_exact = []
for t_ in t:
    y_exact.append(func_exact(t_))
plt.plot(t,y_exact,linestyle='-', linewidth=1)
plt.show()
Ks = range(1,11)
errors = [E(2**(-k)) for k in Ks]
plt.plot(Ks, errors)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('K')
plt.ylabel('Error')
plt.show()

#Q3
# hw 5: page-rank code
# note that you are given a set of code files in pagerank.py
# which you should take as given (you should not edit them unless
# you really have to; treat them like code from a package.)

import numpy as np
from scipy import sparse
import pagerank as pr  # import from given pagerank.py file


# ---------------------------------------------------------------------
# some shell code for the hw [not all functions given]
def pagerank_example():
    """ Solve the 4-node example, recurrent, from class """
    adj, n, names = pr.read_graph_file('small_graph.txt')
    print(f"number of vertices: {n}")  # (leave these prints out of your code)
    print("adjacency list (dictionary):\n", adj)
    print("node names:\n", names)

    # see tiny_example()


def big_example():
    """ Solve the big example (with 10000+ nodes!) """
    adj, n, names = pr.read_graph_file('california.txt')
    for k in range(10):
        print("node {}: {}".format(k, names[k]))

    # note: a full pagerank matrix is *very* large, so
    # you really do have to store it as a sparse matrix.

    # see tiny_example_sparse()


# ---------------------------------------------------------------------
# A bit of example code (calling pr.ranking plus sparse matrices)

def tiny_example():
    """ Pagerank example: three nodes; 0 -> 1, 2 and 1 -> 0, 2 and 2 -> 1 """
    pt = np.array([[0.0, 0.5, 0.5],  # prob = 1/neighbors
                   [0.5, 0, 0.5],
                   [0.0, 1, 0]])
    x = pr.ranking(pt, 50)  # with no teleport!
    print("Stationary distribution (no teleport):")
    print(x)


def tiny_example_sparse():
    """ now consider the above, but add the teleport term (alpha < 1)
        and use a sparse matrix. Obviously this is excessive here, but just
        to illustrate creating a sparse matrix by adding entries."""
    r = [0, 0, 1, 1, 2]  # row indices of points in P^T
    c = [1, 2, 0, 2, 1]  # column indices
    data = [0.5, 0.5, 0.5, 0.5, 1]  # values ( 1/(number of neighbors))

    mat = sparse.coo_matrix((data, (r, c)), shape=(3, 3))
    print(mat.toarray())
    alpha = 0.9
    x = pr.ranking_sparse(mat, alpha, 100)
    print(x)


# -------------------------------------------------------------------
# sparse matrix construction examples: one with manual entry,
# and one with a for loop.
def sparse_example1():
    """ Simple example: a matrix with entries in an x-shaps. """
    n = 5
    # list row/col indices and values: the (r[k],c[k]) entry has value data[k]
    r = [0, 4, 0, 4, 1, 3, 1, 3, 2]  # row indices
    c = [0, 4, 4, 0, 1, 3, 3, 1, 2]  # col indices
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # data (values)

    # function call to make the matrix in COOrdinate form:
    # note that the firt arg. is a tuple: (data, (row indices, col indices))
    mat = sparse.coo_matrix((data, (r, c)), shape=(n, n))
    print(mat.toarray())


def sparse_example2(n):
    """ Example:  build a sparse matrix with 0, 1, 4, ... (n-1)^2
    on the diagonal and 1 1 1 1 ... 1 one entry above the diagonal
    NOTE: you could be more efficient and allocate r,c first here
    """
    r = []
    c = []
    data = []
    for k in range(n - 1):
        r.extend([k, k])  # add entries for (k,k) and (k,k+1)
        c.extend([k, k + 1])
        data.extend([k ** 2, 1])

    r.append(n - 1)  # lower right entry (n,n)
    c.append(n - 1)
    data.append((n - 1) ** 2)

    mat = sparse.coo_matrix((data, (r, c)), shape=(n, n))
    print(mat.toarray())


def pr_matrix(fname, alpha=.9):
    adj, n, names = pr.read_graph_file(fname)
    matrix = np.zeros((n, n), dtype=float)
    # print(matrix.shape)
    for i in range(n):
        portion = 1 / len(adj[i])
        print(portion)
        for j in range(n):
            if j in adj[i]:
                matrix[i, j] = portion
    matrix *= alpha
    return matrix


def myranking(fname, alpha=.9, steps=100):
    mat = pr_matrix(fname, alpha)
    # print(mat)
    adj, n, names = pr.read_graph_file(fname)
    x = pr.ranking(mat, steps)
    indx = np.flip(np.argsort(x))
    lst = []
    for i in indx:
        lst.append(names[i])
    return lst


# print(myranking('small.txt'))

def pr_matrix_sparse(adj):
    n = len(adj.keys())
    r = c = data = []
    for k, v in adj.items():
        for i in v:
            r.append(i)
            c.append(k)
            data.append(1 / len(v))

    mat = sparse.coo_matrix((data, (r, c)), shape=(len(r), len(c)))
    return mat


adj, n, names = pr.read_graph_file('small.txt')
print(pr_matrix_sparse(adj))


def myranking_sparse(fname, alpha=.9, steps=100):
    adj, n, names = pr.read_graph_file(fname)
    mat = pr_matrix_sparse(adj)
    x = pr.ranking_sparse(mat, alpha, steps)
    indx = np.flip(np.argsort(x))
    lst = []
    for i in range(5):
        lst.append(names[i])
    return lst


print(myranking_sparse('california.txt'))
