# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import errors as err
from sklearn.cluster import KMeans

# Reading input files


def read_file(file_name):
    """
    This function takes name of the file and reads it from local directory and
    loads it into a dataframe.Then transposes the dataframe and returns both
    the first and transposed dataframes. It also sets the header for the
    transposed dataframe
    Parameters
    ----------
    file_name : string
        Name of the file to be read into the dataframe.
    Returns
    -------
    A dataframe loaded from the file and it's transpose.
    """

    df = pd.read_csv(file_name)
    df_changed = df.drop(
        columns=["Country Code", "Indicator Name", "Indicator Code"])

    df_changed = df_changed.replace(np.nan, 0)

    # Header
    df_changed = df_changed.rename(columns={'Country Name': 'Year'})
    df_transposed = np.transpose(df_changed)
    # Header setting
    header = df_transposed.iloc[0].values.tolist()
    df_transposed.columns = header
    df_transposed = df_transposed.reset_index()
    df_transposed = df_transposed.rename(columns={"index": "year"})
    df_transposed = df_transposed.iloc[1:]
    df_transposed = df_transposed.dropna()
    # cleaning empty cells
    df_transposed = df_transposed.dropna(axis='columns')
    df_transposed["year"] = df_transposed["year"].str[:4]
    df_transposed["year"] = pd.to_numeric(df_transposed["year"])
    df_transposed["India"] = pd.to_numeric(
        df_transposed["India"])
    df_transposed["United Kingdom"] = pd.to_numeric(
        df_transposed["United Kingdom"])
    df_transposed["year"] = df_transposed["year"].astype(int)
    print(df_transposed['year'])
    return df, df_transposed


df_gdp, df_gdptrans = read_file("GDP Growth.csv")
df_inflation, df_inflationtrans = read_file("Inflation consumer prices.csv")


# Function 1


def curve_fun(t, scale, growth):
    """
    The function uses these parameters to create a curve
    Parameters
    ----------
    t : TYPE
    List of values
    scale : TYPE
    Scale of curve.
    growth : TYPE
    Growth of the curve.
    Returns
    -------
    c : TYPE
    Result
    """
    t = t - 1960
    c = scale * np.exp(growth*t)
    return c


# Fit the curve to the data using curve_fit for India
param, cov = opt.curve_fit(curve_fun, df_gdptrans["year"],
                           df_gdptrans["India"], p0=[4e8, 0.1])

# Calculate the standard deviation of the parameter estimates
sigma = np.sqrt(np.diag(cov))

# Error
low, up = err.err_ranges(df_gdptrans["year"], curve_fun, param, sigma)
df_gdptrans["fit_value"] = curve_fun(df_gdptrans["year"], *param)
print(df_gdptrans["fit_value"])


# Plotting the GDP Growth values for India
plt.figure(figsize=(8, 5))
plt.title("GDP Growth (Annual %) - India")
plt.plot(df_gdptrans["year"], df_gdptrans["India"],
         label="India GDP Growth")
plt.plot(df_gdptrans["year"], df_gdptrans["fit_value"],
         c="red", label="Fit")
plt.fill_between(df_gdptrans["year"], low, up, alpha=0.1)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel("GDP Growth")
plt.savefig("GDP_GROWTH_India.png", dpi=300, bbox_inches='tight')
plt.show()

# Plotting the predicted values for India
plt.figure(figsize=(8, 5))
plt.title("GDP Growth (Annual %) prediction of India ")
pred_year = np.arange(1980, 2030)
pred_bgd = curve_fun(pred_year, *param)
plt.plot(df_gdptrans["year"], df_gdptrans["India"],
         label="India GDP Growth")
plt.plot(pred_year, pred_bgd, label="Predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("GDP Growth")
plt.savefig("GDP_Growth_Predicted_India.png",
            dpi=300, bbox_inches='tight')
plt.show()
##############################################################################
# Fit the curve to the data using curve_fit for UK
param, cov = opt.curve_fit(
    curve_fun, df_gdptrans["year"], df_gdptrans["United Kingdom"],
    p0=[4e8, 0.1])

# Calculate the standard deviation of the parameter estimates
sigma = np.sqrt(np.diag(cov))
# Error
low, up = err.err_ranges(df_gdptrans["year"], curve_fun, param, sigma)
df_gdptrans["fit_value"] = curve_fun(df_gdptrans["year"], *param)
print(df_gdptrans["fit_value"])


# Plotting the GDP Growth values for United Kingdom
plt.figure(figsize=(8, 5))
plt.title("GDP Growth (Annual %) - United Kingdom")
plt.plot(df_gdptrans["year"], df_gdptrans["United Kingdom"],
         label="United Kingdom GDP Growth")
plt.plot(df_gdptrans["year"], df_gdptrans["fit_value"],
         c="red", label="Fit")
plt.fill_between(df_gdptrans["year"], low, up, alpha=0.1)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel("GDP Growth")
plt.savefig("GDP_GROWTH_UK.png", dpi=300, bbox_inches='tight')
plt.show()

# 2: Plotting the predicted values for United Kingdom  GDP Growth
plt.figure(figsize=(8, 5))
plt.title("GDP Growth(Annual %) prediction of United Kingdom")
pred_year = np.arange(1980, 2030)
pred_bgd = curve_fun(pred_year, *param)
plt.plot(df_gdptrans["year"],
         df_gdptrans["United Kingdom"], label="UK GDP Growth")
plt.plot(pred_year, pred_bgd, label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("GDP Growth")
plt.savefig("GDP_Growth_Predicted_United Kingdom.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Fit the curve to the data using curve_fit for UK Prediction Graph
param, cov = opt.curve_fit(
    curve_fun, df_gdptrans["year"], df_gdptrans["United Kingdom"],
    p0=[4e8, 0.1])

# Calculate the standard deviation of the parameter estimates
sigma = np.sqrt(np.diag(cov))

# Error
low, up = err.err_ranges(df_gdptrans["year"], curve_fun, param, sigma)
df_gdptrans["fit_value"] = curve_fun(df_gdptrans["year"], *param)
print(df_gdptrans["fit_value"])
# 2: Plotting the predicted values for United Kingdom  GDP Growth
plt.figure(figsize=(8, 5))
plt.title("GDP Growth(Annual %) prediction of United Kingdom")
pred_year = np.arange(1980, 2030)
pred_bgd = curve_fun(pred_year, *param)
plt.plot(df_gdptrans["year"],
         df_gdptrans["United Kingdom"], label="UK GDP Growth")
plt.plot(pred_year, pred_bgd, label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("GDP Growth")
plt.savefig("GDP_Growth_Predicted_United Kingdom.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Plot Inflation, consumer prices (Annual %) - India
param, cov = opt.curve_fit(curve_fun, df_inflationtrans["year"],
                           df_inflationtrans["India"], p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
low, up = err.err_ranges(df_inflationtrans["year"], curve_fun, param, sigma)
df_inflationtrans["fit_value"] = curve_fun(df_inflationtrans["year"], * param)
plt.figure(figsize=(8, 5))
plt.title("Inflation, consumer prices (Annual %) - India")
plt.plot(df_inflationtrans["year"],
         df_inflationtrans["India"], label="data")
plt.plot(df_inflationtrans["year"],
         df_inflationtrans["fit_value"], c="red", label="fit")
plt.fill_between(df_inflationtrans["year"], low, up, alpha=0.5)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel("Inflation, consumer prices (annual %)")
plt.savefig("Inflation_India.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot Inflation, consumer prices (Annual %) Prediction Values- India
plt.figure(figsize=(8, 5))
plt.title("Inflation, consumer prices (annual %) prediction - India")
pred_year = np.arange(1980, 2030)
pred_ind = curve_fun(pred_year, *param)
plt.plot(df_inflationtrans["year"],
         df_inflationtrans["India"], label="India Inflation")
plt.plot(pred_year, pred_ind, label="Predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Inflation, consumer prices (annual %)")
plt.savefig("Inflation_Prediction_India.png", dpi=300, bbox_inches='tight')
plt.show()


param, cov = opt.curve_fit(curve_fun, df_inflationtrans["year"],
                           df_inflationtrans["United Kingdom"], p0=[4e8,
                                                                    0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
low, up = err.err_ranges(df_inflationtrans["year"], curve_fun, param, sigma)
df_inflationtrans["fit_value"] = curve_fun(df_inflationtrans["year"], * param)
# Plot Inflation, consumer prices (Annual %) - UK
plt.figure(figsize=(8, 5))
plt.title("Inflation, consumer prices (Annual %) - United Kingdom")
plt.plot(df_inflationtrans["year"],
         df_inflationtrans["United Kingdom"], label="United Kingdom")
plt.plot(df_inflationtrans["year"],
         df_inflationtrans["fit_value"], c="Blue", label="fit")
plt.fill_between(df_inflationtrans["year"], low, up, alpha=0.5)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel("Inflation, consumer prices (annual %)")
plt.savefig("Inflation_UK.png", dpi=300, bbox_inches='tight')
plt.show()
# Plot Inflation, consumer prices (Annual %) Prediction Values- UK
plt.figure(figsize=(8, 5))
plt.title("Inflation, consumer prices (annual %) prediction - United Kingdom")
pred_year = np.arange(1980, 2030)
pred_ind = curve_fun(pred_year, *param)
plt.plot(df_inflationtrans["year"],
         df_inflationtrans["United Kingdom"], label="UK Inflation")
plt.plot(pred_year, pred_ind, label="Predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Inflation, consumer prices (annual %)")
plt.savefig("Inflation_Prediction_UK.png", dpi=300, bbox_inches='tight')
plt.show()

# create a new dataframe for India
India = pd.DataFrame()
# add year, GDP growth and inflation data for India to the dataframe
India["year"] = df_gdptrans["year"]
India["GDP Growth"] = df_gdptrans["India"]
India["Inflation"] = df_inflationtrans["India"]
# slice the dataframe
India = India.iloc[1:60, :]
# create a KMeans object with 2 clusters and maximum of 30 iterations
kmean = cluster.KMeans(n_clusters=2, max_iter=30)
# reshape the GDP growth and inflation data into numpy arrays
pt = np.array(India["GDP Growth"]).reshape(-1, 1)
co = np.array(India["Inflation"]).reshape(-1, 1)
# concatenate the GDP growth and inflation data into a single numpy array
cl = np.concatenate((co, pt), axis=1)
nc = 2
kmeans = cluster.KMeans(n_clusters=nc)
# fit the KMeans model to the data
kmeans.fit(cl)
# get the cluster labels for each data point
label = kmeans.labels_
# get the coordinates of the cluster centers
k_center = kmeans.cluster_centers_
col = ['GDP Growth', 'Inflation']
# create dataframe with the GDP growth & inflation data, and the cluster labels
labels = pd.DataFrame(label, columns=['Cluster ID'])
result = pd.DataFrame(cl, columns=col)
# concat result and labels
res = pd.concat((result, labels), axis=1)
# plot the data points with different colors based on their cluster label
plt.figure(figsize=(8.0, 5.0))
plt.title("India GDP Growth vs Inflation, consumer prices")
plt.scatter(res["GDP Growth"], res["Inflation"],
            c=label, cmap="rainbow")
plt.xlabel("GDP growth (annual %)")
plt.ylabel("Inflation, consumer prices")
plt.legend()
# plot the cluster centers as black stars
plt.scatter(k_center[:, 0], k_center[:, 1], marker="*", c="black", s=150)
plt.savefig("Scatter_India.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Plotting Scatter Matrix for India using function


def data_f(country):
    """
    The function uses parameter to create a Scatter Matrix
    Parameters
    ----------

    country : TYPE
    Name of Country.
    Returns
    -------
    None
    """
    pd.plotting.scatter_matrix(country, figsize=(14.0, 12.0))
    plt.tight_layout()
    plt.show()


data_f(India)

# create a new dataframe for United Kingdom
UK = pd.DataFrame()
# add year, GDP growth and inflation data for India to the dataframe
UK["year"] = df_gdptrans["year"]
UK["GDP Growth"] = df_gdptrans["United Kingdom"]
UK["Inflation"] = df_inflationtrans["United Kingdom"]
# slice the dataframe
UK = UK.iloc[1:60, :]
# create a KMeans object with 2 clusters and maximum of 30 iterations
kmean = cluster.KMeans(n_clusters=2, max_iter=30)
# reshape the GDP growth and inflation data into numpy arrays
pt = np.array(UK["GDP Growth"]).reshape(-1, 1)
co = np.array(UK["Inflation"]).reshape(-1, 1)
# concatenate the GDP growth and inflation data into a single numpy array
cl = np.concatenate((co, pt), axis=1)
nc = 2
# fit the KMeans model to the data
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(cl)
# get the cluster labels for each data point
label = kmeans.labels_
# get the coordinates of the cluster centers
k_center = kmeans.cluster_centers_
col = ['GDP Growth', 'Inflation']
# create dataframe with the GDP growth & inflation data, and the cluster labels
labels = pd.DataFrame(label, columns=['Cluster ID'])
result = pd.DataFrame(cl, columns=col)
# concat result and labels
res = pd.concat((result, labels), axis=1)
# plot the data points with different colors based on their cluster label
plt.figure(figsize=(8, 5))
plt.title("United Kingdom GDP Growth vs Inflation, consumer prices")
plt.scatter(res["GDP Growth"], res["Inflation"],
            c=label, cmap="rainbow")
plt.xlabel("GDP growth (annual %)")
plt.ylabel("Inflation, consumer prices")
# plot the cluster centers as black stars
plt.scatter(k_center[:, 0], k_center[:, 1], marker="*",
            c="black", s=150)
plt.legend()
plt.savefig("Scatter_UK.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Plotting Scatter Matrix for UK using function
data_f(UK)
