# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:23:59 2021

@author: Raktim
"""

import numpy as np
from sklearn.preprocessing import FunctionTransformer


transformer = FunctionTransformer(np.log1p)


inverse_transformer = FunctionTransformer(np.expm1)

#X = np.array([[0, 1], [2, 3], [10, 15], [100, 510]])

X = np.array([[0, 1], [2, 3]])

transformed_X=transformer.transform(X)


Orgiginal_get_back =inverse_transformer.transform(transformed_X)
