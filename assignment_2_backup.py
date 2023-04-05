# Importing required packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats


# Creates a list of countries and years to use in the plots
countries = ['China', 'Germany', 'Spain',
             'France', 'United Kingdom', 'Indonesia', 'India',
             'United States']
countries_label = ['China', 'Germany', 'Spain',
                   'France', 'UK', 'Indonesia', 'India',
                   'US']
years = [1990, 1995, 2000, 2005, 2010, 2014]


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
    df_transpose = pd.DataFrame.transpose(df)
    # Header setting
    header = df_transpose.iloc[0].values.tolist()
    df_transpose.columns = header
    return (df, df_transpose)


def clean_df(df_clean):
    """
    Parameters
    ----------
    df : dataframe
        Dataframe that needs to be cleaned and index converted.
    Returns
    -------
    df : dataframe
        dataframe with required columns and index as int.
    """
    # Cleaning the dataframe
    df_clean = df_clean.iloc[1:]
    df_clean = df_clean.iloc[11:58]
    # Converting index to int
    df_clean.index = df_clean.index.astype(int)
    df_clean = df_clean[df_clean.index > 1989]
    # cleaning empty cells
    df_clean = df_clean.dropna(axis='columns')
    return df_clean


def electric_cons_bar(df_energy_total, df_energy_countries):
    """
    The fucntion electric_cons_bar creates bar plots for Electric power
    consumption (kWh per capita) for the eight conturies for the year from
    1990 to 2014 and Graph is also saved in local directory
    Parameters
    ----------
    df_energy_total : This a DataFrame to plot Bar graph forElectric power
    consumption (kWh per capita) for 8 countries. The DataFrame is passed
    as parameter while calling the function.
    df_energy_countries: This a Transposded DataFrame
    Returns
    -------
    None.
    """
    # Cleaning the dataframe
    df_energy_countries = clean_df(df_energy_countries)
    # selecting only required data
    df_energy_time = pd.DataFrame.transpose(df_energy_countries)
    df_energy_subset_time = df_energy_time[years].copy()
    df_energy_subset_time = df_energy_subset_time.loc[df_energy_subset_time.
                                                      index.isin(countries)]
    # plotting the data
    n = len(countries)
    r = np.arange(n)
    width = 0.1
    plt.bar(r-0.3, df_energy_subset_time[1990], color='grey',
            width=width, edgecolor='black', label='1990')
    plt.bar(r-0.2, df_energy_subset_time[1995], color='green',
            width=width, edgecolor='black', label='1995')
    plt.bar(r-0.1, df_energy_subset_time[2000], color='orange',
            width=width, edgecolor='black', label='2000')
    plt.bar(r, df_energy_subset_time[2005], color='red',
            width=width, edgecolor='black', label='2005')
    plt.bar(r+0.1, df_energy_subset_time[2010], color='greenyellow',
            width=width, edgecolor='black', label='2010')
    plt.bar(r+0.2, df_energy_subset_time[2014], color='yellow',
            width=width, edgecolor='black', label='2014')

    plt.xlabel("Countries", fontweight='bold', fontsize=14)
    plt.ylabel("Electricity Consumption", fontweight='bold', fontsize=14)
    plt.xticks(+r, countries_label, rotation=30)
    plt.legend()
    plt.title("Electric power consumption (kWh per capita)",
              fontweight='bold', fontsize=14)
    plt.savefig("Electric_power_consumption.png", dpi=300, bbox_inches='tight')
    plt.show()


def Co2_emission_bar(df_co2emission, df_co2_countries):
    """
    The fucntion Co2_emission_bar creates bar plots for Co2_emission for the
    eight conturies for the year from 1990 to 2014 and Graph is also saved in
    local directory
    Parameters
    ----------
    df_co2emission : This a DataFrame to plot Bar graph for CO2 Emission for
    8 countries for the year 1990 to 2014. The DataFrame is passed as parameter
    while calling the function.
    df_co2_countries: This a Transposded DataFrame
    Returns
    -------
    None.
    """
    # Cleaning the dataframe
    df_co2_countries = clean_df(df_co2_countries)

    # selecting only required data
    df_co2_time = pd.DataFrame.transpose(df_co2_countries)
    df_co2_subset_time = df_co2_time[years].copy()
    df_co2_subset_time = df_co2_subset_time.loc[df_co2_subset_time.index.isin
                                                (countries)]
    n = len(countries)
    r = np.arange(n)
    width = 0.1
    # plotting the data
    plt.bar(r-0.3, df_co2_subset_time[1990], color='aqua',
            width=width, edgecolor='black', label='1990')
    plt.bar(r-0.2, df_co2_subset_time[1995], color='turquoise',
            width=width, edgecolor='black', label='1995')
    plt.bar(r-0.1, df_co2_subset_time[2000], color='steelblue',
            width=width, edgecolor='black', label='2000')
    plt.bar(r, df_co2_subset_time[2005], color='deepskyblue',
            width=width, edgecolor='black', label='2005')
    plt.bar(r+0.1, df_co2_subset_time[2010], color='navy',
            width=width, edgecolor='black', label='2010')
    plt.bar(r+0.2, df_co2_subset_time[2014], color='darkgrey',
            width=width, edgecolor='black', label='2014')
    plt.ticklabel_format(style='plain')
    plt.xlabel("Countries", fontweight='bold', fontsize=14)
    plt.ylabel("CO2 Emission", fontweight='bold', fontsize=14)
    plt.xticks(width+r, countries_label, rotation=30)
    plt.legend()
    plt.title("CO2 Emissions (kt)", fontweight='bold', fontsize=14)
    plt.savefig("Co2 Emissions.png", dpi=300, bbox_inches='tight')
    plt.show()


def population_line_plot(df_Population, df_Population_countries):
    """
    The fucntion population_line_plot creates Line plots for Total Population
    for the eight conturies for the year from 1990 to 2014 and Graph is also
    saved in local directory
    Parameters
    ----------
    df_Population : This a DataFrame to plot Line graph for Total Population
    for 8 countries for the year 1990 to 2014. The DataFrame is passed as
    parameter while calling the function.
    df_Population_countries: This a Transposded DataFrame
    Returns
    -------
    None.
    """

    df_Population_countries = df_Population_countries.iloc[1:]
    df_Population_countries = df_Population_countries.iloc[11:58]
    df_Population_countries.index = df_Population_countries.index.astype(int)
    df_Population_countries = df_Population_countries
    [df_Population_countries.index > 1889]
    df_Population_time = pd.DataFrame.transpose(df_Population_countries)
    df_Population_subset_time = df_Population_time[years].copy()
    df_Population_subset_time = df_Population_subset_time.loc[
        df_Population_subset_time.index.isin(countries)]
    # plotting the data
    plt.figure()
    plt.plot(df_Population_countries.index,
             df_Population_countries["China"], linestyle='dashed')
    plt.plot(df_Population_countries.index,
             df_Population_countries["Germany"], linestyle='dashed')
    plt.plot(df_Population_countries.index,
             df_Population_countries["Spain"], linestyle='dashed')
    plt.plot(df_Population_countries.index,
             df_Population_countries["France"], linestyle='dashed')
    plt.plot(df_Population_countries.index,
             df_Population_countries["United Kingdom"], linestyle='dashed')
    plt.plot(df_Population_countries.index,
             df_Population_countries["Indonesia"], linestyle='dashed')
    plt.plot(df_Population_countries.index,
             df_Population_countries["India"], linestyle='dashed')
    plt.plot(df_Population_countries.index,
             df_Population_countries["United States"], linestyle='dashed')
    plt.xlim(1990, 2014)
    plt.ticklabel_format(style='plain')
    plt.xlabel("Years", fontsize=14, fontweight='bold')
    plt.ylabel("Total Population Values", fontsize=14, fontweight='bold')
    plt.title("Total Population", fontsize=14, fontweight='bold')
    plt.savefig("Total_population.png", dpi=300, bbox_inches='tight')
    plt.legend(countries_label, bbox_to_anchor=(1, 1))
    plt.show()


def electricity_prod_line(df_ele_prod, df_ele_prod_countries):
    """
    The fucntion electricity_prod_line creates Line plots for Electricity
    production from oil, gas, and coal sources for the eight conturies for
    the year from 1990 to 2014 and Graph is also saved in local directory
    Parameters
    ----------
    df_ele_prod : This a DataFrame to plot Line graph for  Electricity
    production from oil, gas, and coal sources for 8 countries for the year
    1990 to 2014. The DataFrame is passed as parameter while calling the
    function.
    df_ele_prod_countries: This a Transposded DataFrame
    Returns
    -------
    None.
    """
    # Cleaning the DataFrame
    df_ele_prod_countries = df_ele_prod_countries.iloc[1:]
    df_ele_prod_countries = df_ele_prod_countries.iloc[11:58]
    df_ele_prod_countries.index = df_ele_prod_countries.index.astype(int)
    df_ele_prod_countries = df_ele_prod_countries[
        df_ele_prod_countries.index > 1889]
    df_ele_prod_time = pd.DataFrame.transpose(df_ele_prod_countries)
    df_ele_prod_subset_time = df_ele_prod_time[years].copy()
    df_ele_prod_subset_time = df_ele_prod_subset_time.loc[
        df_ele_prod_subset_time.index.isin(countries)]
    # plotting the data
    plt.figure()
    plt.plot(df_ele_prod_countries.index, df_ele_prod_countries["China"],
             linestyle='dashdot')
    plt.plot(df_ele_prod_countries.index,
             df_ele_prod_countries["Germany"], linestyle='dashdot')
    plt.plot(df_ele_prod_countries.index,
             df_ele_prod_countries["Spain"], linestyle='dashdot')
    plt.plot(df_ele_prod_countries.index,
             df_ele_prod_countries["France"], linestyle='dashdot')
    plt.plot(df_ele_prod_countries.index,
             df_ele_prod_countries["United Kingdom"], linestyle='dashdot')
    plt.plot(df_ele_prod_countries.index,
             df_ele_prod_countries["Indonesia"], linestyle='dashdot')
    plt.plot(df_ele_prod_countries.index,
             df_ele_prod_countries["India"], linestyle='dashdot')
    plt.plot(df_ele_prod_countries.index,
             df_ele_prod_countries["United States"], linestyle='dashdot')
    plt.xlim(1990, 2014)
    plt.ticklabel_format(style='plain')

    plt.xlabel("Years", fontsize=14, fontweight='bold')
    plt.ylabel("Electricity production", fontsize=14, fontweight='bold')

    plt.title(
        "Electricity production from oil, gas and coal\n sources( % of total)",
        fontsize=14, fontweight='bold')
    plt.savefig("Electricity production.png", dpi=300, bbox_inches='tight')
    plt.legend(countries_label, bbox_to_anchor=(1, 1))
    plt.show()


def country_df(country_name, df_energy_countries, df_co2_countries,
               df_Population_countries, df_ele_prod_countries):
    """
    Creates a dataframe for the country with electricity consumption, 
    co2 emission,Total Population and Electricity Production as columns
    Parameters
    ----------
    country_name : string
        Name of the country to create the dataframe.
    df_energy_countries: This a Transposded DataFrame for Electric Power
    Consumption
    df_co2_countries : This a Transposded DataFrame for CO2 Emission
    df_Population_countries :  This a Transposded DataFrame for Total
    Population
    df_ele_prod_countries : This a Transposded DataFrame for Electricity 
    production from oil, gas, and coal sources
    Returns
    -------
    df_name : dataframe
        Newly created dataframe.
    """
    # creates dataframe name
    df_name = "df_" + country_name
    # creates dataframe
    df_name = pd.concat([df_energy_countries[country_name].astype(float),
                         df_Population_countries[country_name].astype(float),
                         df_co2_countries[country_name].astype(float),
                         df_ele_prod_countries[country_name].astype(float)],
                        axis=1)

    # Gives column names
    df_name.columns.values[0] = "Electric Power Consumption "
    df_name.columns.values[1] = "Population"
    df_name.columns.values[2] = "CO2 Emission"
    df_name.columns.values[3] = "Electricity Prodution"
    return (df_name)


def heatmap(country_name, df_energy_countries, df_co2_countries,
            df_Population_countries, df_ele_prod_countries):
    """
    Creates a correlation heatmap for the country and Dataframes given as
    argument.
    Parameters
    ----------
    country_name : string
        Name of the country to create the heatmap for.
     df_energy_countries: This a Transposded DataFrame for Electric Power
     Consumption
     df_co2_countries : This a Transposded DataFrame for CO2 Emission
     df_Population_countries :  This a Transposded DataFrame for Total
     Population
     df_ele_prod_countries : This a Transposded DataFrame for Electricity
     production from oil, gas, and coal sources
    Returns
    -------
    None.
    """

    df_energy_countries = clean_df(df_energy_countries)
    df_co2_countries = clean_df(df_co2_countries)
    df_Population_countries = df_Population_countries.iloc[1:]
    df_Population_countries = df_Population_countries.iloc[11:58]
    df_Population_countries.index = df_Population_countries.index.astype(int)
    df_Population_countries = df_Population_countries[df_Population_countries.
                                                      index > 1889]
    df_ele_prod_countries = df_ele_prod_countries.iloc[1:]
    df_ele_prod_countries = df_ele_prod_countries.iloc[11:58]
    df_ele_prod_countries.index = df_ele_prod_countries.index.astype(int)
    df_ele_prod_countries = df_ele_prod_countries[df_ele_prod_countries.index
                                                  > 1889]
    # creates dataframe name
    df_name = "df_" + country_name
    # cals function to create dataframe
    df_name = country_df(country_name, df_energy_countries, df_co2_countries,
                         df_Population_countries, df_ele_prod_countries)
    # plots heatmap
    dataplot = sns.heatmap(df_name.corr(), cmap="PuBu", annot=True)
    # saves figure
    filename = country_name + "_heatmap.png"
    plt.title(country_name, fontsize=14, fontweight='bold')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def stat_props():
    """
    The fucntion stat_props calulates the performs several statistical
    calculations on population data from 1990 to 2014
    and print the summary statistics,Average population,
    Median population, Pearsons and Kendall correlation matrix,
    std. deviations, skewness, and kurtosis
    Parameters
    ----------

    Returns
    -------
    None.
    """
    df_Population_stat, df_Population_countries_stat = read_file(
        "Population, total.csv")
    df_Population_stat = df_Population_stat[[
        '1990', '1995', '2000', '2005', '2010', '2014']]
    print(df_Population_stat.describe())
    print("Pearsons correlations")
    print(df_Population_stat.corr())
    print("Kendall correlations")
    print(df_Population_stat.corr(method="kendall"))
    # calculate the Mean and Median
    print("Average population\n", df_Population_stat.mean())
    print("Median population\n", df_Population_stat.median())
    # calculate the skewness,kurtosis and Covariance
    print("std. deviations:\n", df_Population_stat.std())
    print("skewness: \n", df_Population_stat.skew())
    print("kurtosis:\n", df_Population_stat.kurtosis())


# Reading the Data files
df_energy_total, df_energy_countries = read_file(
    "Electric power consumption.csv")
df_co2emission, df_co2_countries = read_file("CO2 emissions (kt).csv")
df_Population, df_Population_countries = read_file("Population, total.csv")
df_ele_prod, df_ele_prod_countries = read_file(
    "Electricity production from oil, gas and coal sources (% of total).csv")


# Calling function to plot the Graphs and to print STATISTICS analysis
electric_cons_bar(df_energy_total, df_energy_countries)
Co2_emission_bar(df_co2emission, df_co2_countries)
population_line_plot(df_Population, df_Population_countries)
electricity_prod_line(df_ele_prod, df_ele_prod_countries)

heatmap("United States", df_energy_countries, df_co2_countries,
        df_Population_countries, df_ele_prod_countries)
heatmap("China", df_energy_countries, df_co2_countries,
        df_Population_countries, df_ele_prod_countries)
heatmap("Germany", df_energy_countries, df_co2_countries,
        df_Population_countries, df_ele_prod_countries)

stat_props()
