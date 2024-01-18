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
import scipy.optimize as opt
import errors as err


def read_world_bank_csv(filename, start_year=1990, end_year=2020, countries=None):
    # read csv using pandas
    wb_df = pd.read_csv(filename, skiprows=3, iterator=False)

    # clean na data, remove columns
    wb_df.dropna(axis=1)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(start_year, (end_year + 1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select specific countries and years
    if countries:
        df_country_index = wb_df.loc[wb_df["Country Name"].isin(countries), all_cols_list]
    else:
        df_country_index = wb_df.loc[:, all_cols_list]

    # make the country as an index and then drop the column as it becomes the index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns to integer
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index)

    # return the two dataframes: year as an index and country as an index
    return df_year_index, df_country_index

def one_silhouette(xy, n):
    """ Calculates silhouette score for n clusters """
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)  # fit done on x, y pairs
    labels = kmeans.labels_
    score = (skmet.silhouette_score(xy, labels))
    return score

def poly(x, a, b, c, d):
    """ Calulates polynominal"""
    
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3 
    
    return f

def cluster_graph(df_cluster):
    
# Normalize the data for clustering
    df_norm, df_min, df_max = ct.scaler(df_cluster)


# Number of clusters for KMeans
    ncluster = 3

# set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=min(ncluster, len(df_norm)), n_init=20)
# Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)  # fit done on x, y pairs

# Assign cluster labels to the original data
    df_cluster['cluster_label'] = kmeans.labels_

# Extract the estimated cluster centers and convert to original scales
    cen = kmeans.cluster_centers_
    cen = ct.backscale(cen, df_min, df_max)

    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]


# Plot the clusters
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Subplot for 1991
    axes[0].scatter(df_cluster["green_house"], df_cluster["gdp"], 10, df_cluster['cluster_label'], cmap="Paired")
    axes[0].scatter(xkmeans, ykmeans, 45, "k", marker="d", label="Cluster Centers")
    axes[0].set_xlabel("Greenhouse Gas Emissions in 1991 (x 0.000)")
    axes[0].set_ylabel("GDP in 1991 (x 10,000)")
    axes[0].set_title("Country cluster based on Greenhouse and GDP(1990)")
    axes[0].legend()
    axes[0].grid(True)

    df_norm_2020, df_min_2020, df_max_2020 = ct.scaler(df_clus[["green_house", "gdp"]])
    kmeans_2020 = cluster.KMeans(n_clusters=min(ncluster, len(df_norm_2020)), n_init=20)
    kmeans_2020.fit(df_norm_2020)
    df_clus['cluster_label_2020'] = kmeans_2020.labels_
    cen_2020 = kmeans_2020.cluster_centers_
    cen_2020 = ct.backscale(cen_2020, df_min_2020, df_max_2020)
    xkmeans_2020 = cen_2020[:, 0]
    ykmeans_2020 = cen_2020[:, 1]



    axes[1].scatter(df_clus["green_house"], df_clus["gdp"], 10, df_clus['cluster_label_2020'], cmap="Paired")
    axes[1].scatter(xkmeans_2020, ykmeans_2020, 45, "k", marker="d", label="Cluster Centers")
    axes[1].set_xlabel("Greenhouse Gas Emissions in 2020 (x 0.000)")
    axes[1].set_ylabel("GDP in 2020 (x 10,000)")
    axes[1].set_title("Country cluster based on Greenhouse and GDP(2020)")
    axes[1].legend()
    axes[1].grid(True)

def fitting(green_house_yw,country):
    #####################fitting############################


    df_fit = pd.DataFrame()
    #print(df_fit)
    df_fit["green_house"] = green_house_yw[country].copy()
    df_fit["gdp"] = gdp_yw[country]
    df_fit =df_fit.dropna()

    plt.figure()

    df_fit["Year"] = df_fit.index
    #print(df_fit)

    param, covar = opt.curve_fit(poly, df_fit["Year"],df_fit["green_house"])
    df_fit["fit"] = poly(df_fit["Year"], *param)

    df_fit.plot("Year",["green_house", "fit"])
    plt.xlabel("Year")
    plt.ylabel("Greenhouse Gases(%)")
    plt.title("Greenhouse Forecast("+country+")")
    plt.legend()
    plt.grid()

    ############Forecast###############
    year = np.arange(1990, 2030)
    forecast = poly(year, *param)
    sigma = err.error_prop(year, poly, param, covar)
    low = forecast - sigma
    up = forecast + sigma

    df_fit["fit"] = poly(df_fit["Year"], *param)

    plt.figure()
    plt.plot(df_fit["Year"], df_fit["green_house"], label="Green_house")
    plt.plot(year, forecast, label="forecast")
    plt.ylabel("Greenhouse Gases(%)")
    plt.title("Greenhouse Forecast("+country+")")
    plt.legend()
    plt.grid(True)
    # plot uncertainty range
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    
    
# read csv files and get the dataframes
green_house_yw, green_house_cw = read_world_bank_csv("API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_6299833.csv")
gdp_yw, gdp_cw = read_world_bank_csv("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv")

# Create a DataFrame for clustering
df_cluster = pd.DataFrame()
df_clus = pd.DataFrame()
df_clus["green_house"] = green_house_cw[2020]
df_clus["gdp"] = gdp_cw[2020]
df_cluster["green_house"] = green_house_cw[1990]
df_cluster["gdp"] = gdp_cw[1990]
df_clus = df_clus.dropna()
df_cluster = df_cluster.dropna()

cluster_graph(df_cluster)
fitting(green_house_yw,"India")
fitting(green_house_yw,"China")
fitting(green_house_yw,"Brazil")


# show all plots
plt.show()
