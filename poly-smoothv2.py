#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt

import itertools


def polyfit2d(x, y, f, deg):
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)


import os
import pandas as pd

table_size = 32
deg = np.array([2,5])
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

w = polyfit2d(x, y, z, deg)

# Generate x', y', z'
n_ = 2048
x_, y_ = np.meshgrid(np.linspace(x.min(), x.max(), n_),
                      np.linspace(y.min(), y.max(), n_))
z_ = np.zeros((n_, n_))
for i in range(n_):
    z_[i, :] = polynomial.polyval2d(x_[i, :], y_[i, :], w)

# Plot
plt.imshow(z_, aspect='auto', extent=(x_.min(), x_.max(), y_.max(), y_.min()))
plt.scatter(x, y, c=z)
plt.show()
