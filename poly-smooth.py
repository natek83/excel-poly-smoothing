#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import itertools


# w = (Phi^T Phi)^{-1} Phi^T t
# where Phi_{k, j + i (m_2 + 1)} = x_k^i y_k^j,
#       t_k = z_k,
#           i = 0, 1, ..., m_1,
#           j = 0, 1, ..., m_2,
#           k = 0, 1, ..., n - 1
def polyfit2d(x, y, z, m_1, m_2):
    # Generate Phi by setting Phi as x^i y^j
    nrows = x.size
    ncols = (m_1 + 1) * (m_2 + 1)
    Phi = np.zeros((nrows, ncols))
    ij = itertools.product(range(m_1 + 1), range(m_2 + 1))
    for h, (i, j) in enumerate(ij):
        Phi[:, h] = x ** i * y ** j
    # Generate t by setting t as Z
    t = z
    # Generate w by solving (Phi^T Phi) w = Phi^T t
    w = np.linalg.solve(Phi.T.dot(Phi), (Phi.T.dot(t)))
    return w


# t' = Phi' w
# where Phi'_{k, j + i (m_2 + 1)} = x'_k^i y'_k^j
#       t'_k = z'_k,
#           i = 0, 1, ..., m_1,
#           j = 0, 1, ..., m_2,
#           k = 0, 1, ..., n' - 1
def polyval2d(x_, y_, w, m_1, m_2):
    # Generate Phi' by setting Phi' as x'^i y'^j
    nrows = x_.size
    ncols = (m_1 + 1) * (m_2 + 1)
    Phi_ = np.zeros((nrows, ncols))
    ij = itertools.product(range(m_1 + 1), range(m_2 + 1))
    for h, (i, j) in enumerate(ij):
        Phi_[:, h] = x_ ** i * y_ ** j
    # Generate t' by setting t' as Phi' w
    t_ = Phi_.dot(w)
    # Generate z_ by setting z_ as t_
    z_ = t_
    return z_



import os
import pandas as pd

table_size = 32
x_deg = 2
y_deg = 5
# strength = 3

fname = os.path.join("2 7 Hydra LTT Conversion (v1.1).xlsx")
fuel = pd.read_excel(fname,sheet_name=1,index_col=0,nrows= table_size)

x_s = fuel.columns
y_s = fuel.index
xi = []
yi = []
x = []
y = []
z = []

for i in range(0, table_size):
    xi.append(int(x_s[i].replace('rpm','')))
    
for i in range(0, table_size):
    yi.append(int(y_s[i].replace('kPa','')))
    
for i in range(0, table_size):
    for j in range(0, table_size):
        x.append(xi[i])

for i in range(0, table_size):
    for j in range(0, table_size):
        y.append(yi[j])

for i in range(0, table_size):
    for j in range(0, table_size):
        z.append(fuel.loc[y_s[j],x_s[i]])

x = np.array(x)
y = np.array(y)
z = np.array(z)

w = polyfit2d(x, y, z, m_1=x_deg, m_2=y_deg)

# Generate x', y', z'
n_ = 2048
x_, y_ = np.meshgrid(np.linspace(x.min(), x.max(), n_),
                      np.linspace(y.min(), y.max(), n_))
z_ = np.zeros((n_, n_))
for i in range(n_):
    z_[i, :] = polyval2d(x_[i, :], y_[i, :], w, m_1=x_deg, m_2=y_deg)

# Plot
plt.imshow(z_, aspect='auto', extent=(x_.min(), x_.max(), y_.max(), y_.min()))
plt.scatter(x, y, c=z)
plt.show()
