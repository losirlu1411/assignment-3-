# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:30:15 2024

@author: losir
"""

import pandas as pd
import numpy as np

import sklearn.cluster as cluster
import sklearn.metrics as skmet

import matplotlib.pyplot as plt
import cluster_tools as ct


    # Load data from the CSV file
    data1 = pd.read_csv(filename, skiprows=3)

    # Set the index and select specific countries and years
    data1.index = data1.iloc[:, 0]
    data1 = data1.iloc[:, 1:]
    countries = ["China", "United States","India", "Brazil","South Africa","Japan", "Germany"]
    years = np.arange(1990,2020).astype("str")
    data1 = data1.loc[countries, years]

# Transpose the data for easier plotting
    data1_t = data1.T
    data1_t.index = data1_t.index.astype(int)
    return data1, data1_t

data, data_t = read_data("API_EN.ATM.METH.KT.CE_DS2_en_csv_v2_5995564.csv")
 