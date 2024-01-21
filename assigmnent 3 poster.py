# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:30:15 2024

@author: losir
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import cluster
import sklearn.metrics as skmet
import errors as err
import cluster_tools as ct


def read_world_bank_csv(filename, start_year=1990, end_year=2020, countries=None):
    """
    

    Parameters
    ----------
    filename : it is a csv file
        It contains data of the country name and year.
    start_year : It is an integer.
        DESCRIPTION. The default is 1990.
    end_year : It is an integer
        DESCRIPTION. The default is 2020.
    countries : It is an integer
        DESCRIPTION. it contains countries.

    Returns
    -------
    df_year_index : it is a data frame
        It contains year as index.
    df_country_index : It is a  data frame
        It contains countries as index.

    """
    # read csv using pandas
    wb_df = pd.read_csv(filename, skiprows=3, iterator=False)

    # clean na data, remove columns
    wb_df.dropna(axis=1)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(start_year, (end_year + 1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select specific countries and years
    if countries:
        df_country_index = wb_df.loc[wb_df["Country Name"].isin(
            countries), all_cols_list]
    else:
        df_country_index = wb_df.loc[:, all_cols_list]

    # make the country as an index and then drop the column as it becomes the index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns to integer
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index)

    return df_year_index, df_country_index


def one_silhouette(xy_variable, value):
    """ Calculates silhouette score for n clusters """
    kmeans = cluster.KMeans(n_clusters=value, n_init=20)
    kmeans.fit(xy_variable)  # fit done on x, y pairs
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy_variable, labels)
    return score


def poly(x_variable, a_cons, b_cons, c_cons, d_cons):
    """ Calulates polynominal"""

    x_dot = x_variable - 1990
    fun = a_cons + b_cons*x_dot + c_cons*x_dot**2 + d_cons*x_dot**3

    return fun


def cluster_graph(df_data, df_data2):    
    """
    

    Parameters
    ----------
    df_data : It is a datatype.
        It contains data of clusters.
    df_data2 :It is a datatype.
        It contains data of clusters.
    Returns
    -------
    None.

    """

    # Normalize the data for clustering
    df_norm, df_min, df_max = ct.scaler(df_data)


    # Number of clusters for KMeans
    ncluster = 2

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=min(ncluster, len(df_norm)), n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit_predict(df_norm)  # fit done on x, y pairs

    # Assign cluster labels to the original data
    df_data['cluster_label'] = kmeans.labels_
    df_data.to_csv("cluster_label_1990.csv")
    # Extract the estimated cluster centers and convert to original scales
    cen = kmeans.cluster_centers_
    cen = ct.backscale(cen, df_min, df_max)

    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]


    # Plot the clusters
    plt.figure()
    _, axes = plt.subplots(1, 2, figsize=(16, 8))

   # Subplot for 1990
    axes[0].scatter(df_data["green_house"], df_data["gdp"],
                    10, df_data['cluster_label'], cmap="Paired")
    axes[0].scatter(xkmeans, ykmeans, 45, "k",
                    marker="d", label="Cluster Centers")
    axes[0].set_xlabel(
        "Greenhouse Gas Emissions in 1990 (x 0.000)", fontsize=18)
    axes[0].set_ylabel("GDP in 1990 (x 10,000)", fontsize=20)
    axes[0].set_title(
        "Country cluster based on Greenhouse and GDP(1990)", fontsize=15)
    axes[0].legend()
    axes[0].grid(True)

    df_norm_2020, df_min_2020, df_max_2020 = ct.scaler(
        df_data2[["green_house", "gdp"]])
    kmeans_2020 = cluster.KMeans(n_clusters=min(
        ncluster, len(df_norm_2020)), n_init=20)
    kmeans_2020.fit(df_norm_2020)
    df_data2['cluster_label_2020'] = kmeans_2020.labels_
    df_data2.to_csv("cluster_label_2020.csv")
    cen_2020 = kmeans_2020.cluster_centers_
    cen_2020 = ct.backscale(cen_2020, df_min_2020, df_max_2020)
    xkmeans_2020 = cen_2020[:, 0]
    ykmeans_2020 = cen_2020[:, 1]

    axes[1].scatter(df_data2["green_house"], df_data2["gdp"], 10,
                    df_data2['cluster_label_2020'], cmap="Paired")
    axes[1].scatter(xkmeans_2020, ykmeans_2020, 45, "k",
                    marker="d", label="Cluster Centers")
    axes[1].set_xlabel(
        "Greenhouse Gas Emissions in 2020 (x 0.000)", fontsize=18)
    axes[1].set_ylabel("GDP in 2020 (x 10,000)", fontsize=20)
    axes[1].set_title(
        "Countries cluster based on Greenhouse and GDP(2020)", fontsize=15)
    axes[1].legend()
    axes[1].grid(True)


def fitting(df_fitting,df_gdp_yw, country):
    """
    

    Parameters
    ----------
    df_fitting : it is a data frame
        It contains the data of fitting.
    df_gdp_yw : t is a data frame
        It contains the data of gdp.
    country : It is a variable
        It contain data of country.

    Returns
    -------
    None.

    """


    #####################fitting############################

    cr_year = pd.DataFrame()
    # print(df_fit)
    cr_year["green_house"] = df_fitting[country].copy()
    cr_year["gdp"] = df_gdp_yw[country]
    cr_year.dropna(inplace=True)



    cr_year["Year"] = cr_year.index
    # print(df_fit)

    param, covar = opt.curve_fit(poly, cr_year["Year"], cr_year["green_house"])
    cr_year["fit"] = poly(cr_year["Year"], *param)

    cr_year.plot("Year", ["green_house", "fit"])

    plt.xlabel("Year")
    plt.ylabel("Greenhouse Gases")
    plt.title("Greenhouse Forecast("+country+")")
    plt.legend()
    plt.grid()
    plt.savefig(".jpg", dpi=300, bbox_inches="tight")
    ############Forecast###############
    year = np.arange(1990, 2025)
    forecast = poly(year, *param)
    sigma = err.error_prop(year, poly, param, covar)
    low_boundry = forecast - sigma
    up_bountry = forecast + sigma

    cr_year["fit"] = poly(cr_year["Year"], *param)

    plt.figure()
    plt.plot(cr_year["Year"], cr_year["green_house"], label="Green_house")
    plt.plot(year, forecast, label="forecast")
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Greenhouse Gases", fontsize=20)
    plt.title("Greenhouse Forecast("+country+")", fontsize=20)
    plt.legend()
    plt.grid(True)
    # plot uncertainty range
    plt.fill_between(year, low_boundry, up_bountry, color="yellow", alpha=0.7)


# read csv files and get the dataframes
green_house_yw, green_house_cw = read_world_bank_csv(
    "API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_6299833.csv")
gdp_yw, gdp_cw = read_world_bank_csv(
    "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv")

# Create a DataFrame for clustering
df_cluster_2020 = pd.DataFrame()
df_cluster_1990 = pd.DataFrame()
df_cluster_1990["green_house"] = green_house_cw[2020]
df_cluster_1990["gdp"] = gdp_cw[2020]
df_cluster_2020["green_house"] = green_house_cw[1990]
df_cluster_2020["gdp"] = gdp_cw[1990]
df_cluster_1990 = df_cluster_1990.dropna()
df_cluster_2020 = df_cluster_2020.dropna()

cluster_graph(df_cluster_2020, df_cluster_1990)
plt.savefig("cluster.png", dpi=300, bbox_inches="tight")
fitting(green_house_yw, gdp_yw, "India")
plt.savefig("India.png", dpi=300, bbox_inches="tight")
fitting(green_house_yw, gdp_yw, "China")
plt.savefig("China.png", dpi=300, bbox_inches="tight")
fitting(green_house_yw, gdp_yw, "Brazil")
plt.savefig("Brazil.png", dpi=300, bbox_inches="tight")

# show all plots
plt.show()