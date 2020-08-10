#Q1
#a) if elements of list a and list b are same then a == b will be True

#for example below code block results in True

#( If b =[3, 2, 1] then a == b will be false )

#That means elements at particular index should also be same in both lists for == to
#return True

#a = [1, 2, 3]
#b = [1, 2, 3]
#a == b
#b) Merge
#Sort


def mergeSort(arr):
    if len(arr) > 1:
        # get mid index of array
        mid = len(arr) // 2
        # divide array into two halfs
        L = arr[:mid]
        R = arr[mid:]
        # sort left half
        mergeSort(L)
        # sort right half
        mergeSort(R)
        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        # Checking if there are any left elements which were not added
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


# main method
if __name__ == '__main__':
    arr = [20, 19, 34, 56, 1, 2, 134]
    print("Sorting using in built method")
    print(sorted(arr))
    print("Original array")
    print(arr)
    mergeSort(arr)
    print("Array after sorting via Merge sort")
    print(arr)
#output

#Sorting
#using in built
#method
#[1, 2, 19, 20, 34, 56, 134]
#Original
#array
#[20, 19, 34, 56, 1, 2, 134]
#Array after sorting via Merge sort
#[1, 2, 19, 20, 34, 56, 134
#[20, 19, 34, 56, 1, 2, 134]
# / \
 #[20, 19, 34][56, 1, 2, 134]
 #/      \ / \
    #[20][19, 34][56, 1][2, 134]
    #| /    \ /     \ / \
   # | [19][34][56][1][2][134]
     # | | | | | | |
     # |      \ /      \ /     \ /
#[20][19, 34][1, 56][2, 134]
#| | | |
#\ /             \ /
#[19, 20, 34][1, 2, 56, 134]
#| |
#\ /
#[1, 2, 19, 20, 34, 56, 134]
#c)

import random

# setting seed value to 42
random.seed(42)
# let array size be 20
n = 20
arr = []
for i in range(n):
    # randomly generating element value in range of 500 and adding to list
    arr.append(random.randint(1, 500))
print(arr)
OUTPUT

#[328, 58, 13, 380, 141, 126, 115, 72, 378, 53, 347, 380, 457, 280, 45, 303, 217, 17, 16, 48]
#As
#long as seddvalue is 42
#same
#random
#elements
#are
#generated and list
#will be same(That is the main use of seed, to generate same random values at particular seed value).

#d)

import random
from time import perf_counter

# setting seed value to 42
random.seed(42)


def mergeSort(arr):
    temp = arr
    if len(arr) > 1:
        # get mid index of array
        mid = len(arr) // 2
        # divide array into two halfs
        L = arr[:mid]
        R = arr[mid:]
        # sort left half
        mergeSort(L)
        # sort right half
        mergeSort(R)
        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        # Checking if there are any left elements which were not added
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


# main method
if __name__ == '__main__':
    # let array size be 30
    n = 100
    arr = []
    for i in range(n):
        # randomly generating element value in range of 500 and adding to list
        arr.append(random.randint(1, 500))
    print(arr)
    i = 0
    start = perf_counter()  # start time counter
    # perform inbuilt sort
    while i < 1000:
        sorted(arr)
        i += 1
    # stop counter
    stop = perf_counter()
    # print time taken (you can print average by dividing it by 1000
    print("time using inbuilt", stop - start)
    print("Original array")
    print(arr)
    i = 0
    # since merge sort changes array in place i pass temp array so that random array is sorted every time
    temp = arr
    start = perf_counter()
    while i < 1000:
        mergeSort(temp)
        i += 1
        temp = arr
    stop = perf_counter()
    print("time using merge sort", stop - start)
    print(arr)

#


def integ1(fun, a, b, tol=1e-8, depth=0, endpt=[]):
    """ recursive adaptive scheme. Tracks left endpoint of sub-intervals
        used in the shared list `endpt' for testing."""

    stack = []
    stack.append((a, b, tol, depth, endpt))
    while len(stack):
        a, b, tol, depth, endpt = stack.pop()
        c = 0.5 * (a + b)
        a1 = simp(fun, a, b)
        a1_fin = a1
        a2 = simp(fun, a, c) + simp(fun, c, b)
        err = (16 / 15) * abs(a1 - a2)
        print (depth, a, b, tol)
        if err < tol or depth > MAX_DEPTH:
            endpt.append(a)
            endpt_final = endpt
            continue
        #             return a1, endpt
        stack.append((a, c, tol / 2, depth + 1, endpt))
        stack.append((c, b, tol / 2, depth + 1, endpt))
    #         a1 = integ(fun, a, c, tol/2, depth+1)[0]
    #         a2 = integ(fun, c, b, tol/2, depth+1)[0]
    print(a1_fin, endpt_final)
    return a1_fin, endpt_final


# Q2
# adaptive integration example using Simpson's rule and recursion.
import inspect
import numpy as np
import matplotlib.pyplot as plt

MAX_DEPTH = 10  # max recursion depth


def simp_equal(f, a, b, n):
    """ simpson's rule, n sub-intervals (equally spaced) """
    h = (b - a) / n
    total = f(a)
    for k in range(1, n // 2 + 1):
        c = a + (2 * k - 1) * h
        total += 4 * f(c)
        total += 2 * f(c + h)
    return (total - f(b)) * h / 3


def simp(fun, a, b):
    """ simpson's rule, two sub-intervals for integ """
    h = 0.5 * (b - a)
    c = 0.5 * (a + b)
    return (h / 3) * (fun(a) + 4 * fun(c) + fun(b))


def integ(fun, a, b, tol=1e-8, depth=0, endpt=[]):
    """ recursive adaptive scheme. Tracks left endpoint of sub-intervals
        used in the shared list `endpt' for testing."""
    #     print (depth, a, b, tol)
    c = 0.5 * (a + b)
    a1 = simp(fun, a, b)
    a2 = simp(fun, a, c) + simp(fun, c, b)

    err = (16 / 15) * abs(a1 - a2)

    if err < tol or depth > MAX_DEPTH:
        endpt.append(a)
        return a1, endpt

    a1 = integ(fun, a, c, tol / 2, depth + 1)[0]
    a2 = integ(fun, c, b, tol / 2, depth + 1)[0]
    return a1 + a2, endpt


def integ1(fun, a, b, tol=1e-8, depth=0, endpt=[]):
    """ recursive adaptive scheme. Tracks left endpoint of sub-intervals
        used in the shared list `endpt' for testing."""

    stack = []
    stack.append((a, b, tol, depth, endpt))
    area = 0
    while len(stack):
        a, b, tol, depth, endpt = stack.pop()
        c = 0.5 * (a + b)
        a1 = simp(fun, a, b)
        a2 = simp(fun, a, c) + simp(fun, c, b)
        err = (16 / 15) * abs(a1 - a2)
        #         print (depth, a, b, tol)
        if err < tol or depth > MAX_DEPTH:
            endpt.append(a)
            area += a1
            #             print('-->', a1)
            continue
        #             return a1, endpt
        stack.append((a, c, tol / 2, depth + 1, endpt))
        stack.append((c, b, tol / 2, depth + 1, endpt))
    #         a1 = integ(fun, a, c, tol/2, depth+1)[0]
    #         a2 = integ(fun, c, b, tol/2, depth+1)[0]

    #     print('Area', area)
    return area, endpt


def f(x):
    return np.sin(1.0 / x)


if __name__ == '__main__':
    exact = 0.51301273999140998  # from wolfram alpha
    a = 0.1
    b = 1
    tol = 2e-5

    approx_rec, left_rec = integ(f, a, b, tol)
    approx_, left_ = integ1(f, a, b, tol)

    print(sorted(left) == sorted(left_rec))
    err = abs(approx_ - exact)

    # get equally spaced approx. with the same err. bound
    err_s = 1
    n = 2
    while err_s > tol:
        n *= 2
        s = simp_equal(f, a, b, n)
        err_s = abs(s - exact)

    print('Adapt: err = {:.2e}, {} intervals'.format(err, len(left_)))
    print('Equal: err = {:.2e}, {} intervals'.format(err_s, n))

    x = np.linspace(a, b, 100)
    left = np.array(left_)  # convert to numpy arrray for vectorized math

    plt.figure(figsize=(3, 2.5))
    plt.plot(left, f(left), '.b')  # plot the left sub-interval points
    plt.plot(x, f(x), '--k')  # plot the function
    plt.xlabel('$x$')
    plt.show()


# Q3
import random


class BNode:
    """ Binary search tree node class, with parent for convenience"""

    def __init__(self, data, left=None, right=None, parent=None):
        self.data = data
        self.children = [left, right]
        self.parent = parent

    def __repr__(self):
        return "Node({})".format(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __gt__(self, other):
        return self.data > other.data


class BTree:
    def __init__(self, root):
        self.root = root

    def leaves(self):
        if self.root.children[0]:
            self.root.children[0].leaves()
        print(self.data),
        if self.root.children[1]:
            self.root.children[1].leaves()

    def depth(self):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.solve(self.root)

    def solve(self, root, depth=0):
        if root is None:
            return depth
        return max(self.solve(root.children[0], depth + 1), self.solve(root.children[1], depth + 1))

    def insert(self, val):
        """ insert a new leaf node with data val"""
        node = self.root
        while node:
            p = node
            side = val > node.data
            node = node.children[side]

        p.children[side] = BNode(val, None, None, p)  # new leaf, parent p

    def find(self, val):
        """ find a node with data val """
        n = self.root
        while n and n.data != val:
            n = n.children[val > n.data]

        return n  # returns None if not found

    def remove(self, val):
        """ removes a node with data val """
        n = self.find(val)
        if not n:
            print("node not found!")
            return
        else:
            self.delete(n)

    def delete(self, n):
        """ deletes the specified node (must be in the tree)"""
        left = n.children[0]
        right = n.children[1]

        # trivial cases: root node, at most one child
        if n == self.root and (not left or not right):
            if right:
                self.root = right
                right.parent = None
            elif left:
                self.root = left
                left.parent = None
            else:
                self.root = None
            return

        if not left and not right:  # leaf
            side = n.data > n.parent.data  # side to follow for parent -> child
            n.parent.children[side] = None  # cut off the node
        elif not left:  # no left branch
            n.parent.children[1] = right
            right.parent = n.parent
        elif not right:  # no right branch
            n.parent.children[0] = left
            left.parent = n.parent
        else:  # has a left branch
            while left.children[1]:  # find largest predecessor
                left = left.children[1]

            n.data = left.data  # move up the predecessor
            self.delete(left)  # ... recursively remove old node

    def __repr__(self):
        """print the tree (badly), with each depth on one line.
           Uses * to denote None (no child for a node)"""
        level = 0
        q = [(self.root, level)]
        rep = ""
        while q:
            n, level = q.pop(0)
            if not n:
                rep += "*"
            else:
                rep += str(n.data)
                q.append((n.children[0], level + 1))
                q.append((n.children[1], level + 1))
            rep += " "
            if q and q[0][1] > level:
                rep += "\n"

        return rep


def example(n):
    """ Build an example tree with n random ints, return the tree/values """
    tree = BTree(BNode(500))
    vals = [500]
    for k in range(n):
        x = random.randint(1, 1000)
        tree.insert(x)
        vals.append(x)

    return tree, vals


if __name__ == "__main__":
    depths = []
    for i in range(1000):
        t, v = example(100)
        depths.append(t.depth())
    print(sum(depths) / len(depths))