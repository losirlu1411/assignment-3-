# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:30:15 2024

@author: losir
"""

import numpy as np
import pandas as pd
import cluster_tools as ct
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt
import errors as err

def read_world_bank_csv(filename):

    # set year range and country list to filter the dataset
    start_from_yeart = 1970
    end_to_year = 2020
    #countrie_list = ["Brazil", "Indonesia", "Russian Federation", "Argentina",
     #                "Paraguay", "China", "Nigeria","India"]

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)

    # clean na data, remove columns
    #wb_df.dropna(axis=1)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select only specific countries and years
    df_country_index = wb_df.loc[
       # wb_df["Country Name"].isin(countrie_list),
       :, all_cols_list]

    # make the country as index and then drop column as it becomes index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as interger
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index).copy()

    # return the two dataframes year as index and country as index
    return df_year_index, df_country_index

def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    
    # makes it easier to get a guess for initial parameters
    t = t - 1990
    
    f = n0 * np.exp(g*t)
    
    return f

def logistic(t, n0, g):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t )))
    
    return f


def poly(x, a, b, c, d):
    """ Calulates polynominal"""
    
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3 
    
    return f

###### Main Function ################

# read csv files and get the dataframs

green_house_yw, green_house_cw = \
    read_world_bank_csv("API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_6299833.csv")

gdp_yw, gdp_cw = \
    read_world_bank_csv("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv")



###############Clustering   ######################################

df_cluster = pd.DataFrame()
#df_cluster["co2"] = co2_data_yw["India"]
#df_cluster["agri"] = agri_lnd_yw["India"]

df_cluster["green_house"] = green_house_cw[1991]
df_cluster["gdp"] = gdp_cw[1991]
#df_cluster["co2_1"] = green_house_cw[2020]
#print(df_cluster)
df_cluster = df_cluster.dropna()

#print(df_cluster)

df_norm, df_min, df_max = ct.scaler(df_cluster)


# calculate silhouette score for 2 to 10 clusters
#for ic in range(2, 11):
#    score = one_silhoutte(df_cluster, ic)
#    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   # allow for minus signs
    

ncluster = 3


# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs


labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = ct.backscale(cen, df_min, df_max)

xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

# extract x and y values of data points
x = df_cluster["green_house"]
y = df_cluster["gdp"]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
cm = plt.colormaps["Paired"]
plt.scatter(x, y, 10, labels, marker="o", cmap=cm)
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.scatter(xkmeans, ykmeans, 45, "y", marker="+")
plt.xlabel("height")
plt.ylabel("width")

#####################fitting############################


df_fit = pd.DataFrame()
#print(df_fit)
df_fit["green_house"] = green_house_yw["India"].copy()
df_fit["gdp"] = gdp_yw["India"]
df_fit =df_fit.dropna()
plt.figure()
df_fit["Year"] = df_fit.index
print(df_fit)

param, covar = opt.curve_fit(poly, df_fit["Year"],df_fit["green_house"])
df_fit["fit"] = poly(df_fit["Year"], *param)

df_fit.plot("Year",["green_house", "fit"])


############Forcast###############
year = np.arange(1990, 2025)
forecast = poly(year, *param)
sigma = err.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma

df_fit["fit"] = poly(df_fit["Year"], *param)

plt.figure()
plt.plot(df_fit["Year"], df_fit["green_house"], label="GDP")
plt.plot(year, forecast, label="forecast")

# plot uncertainty range
plt.fill_between(year, low, up, color="yellow", alpha=0.7)


# show all plots
plt.show()


    