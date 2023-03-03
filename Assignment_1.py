# Importing required packages
import matplotlib.pyplot as plt
import pandas as pd

"""
This program produces  different plots (Line, Stacked Bar and Pie chart)
to compare and analyse the three different datasets using pyplot with three
different functions.
"""


def line_plot(linedata):
    """
    The fucntion line_plot creates line plots for Milk used in the
    production of milk products(2011-22)in the UK and Graph is also saved in
    local directory
    Parameters
    ----------
    linedata : This a DataFrame to plot line graph for Milk used in the
    production of milk products(2011-22)in the UK. The DataFrame is passed
    as parameter while calling the function.

    Returns
    -------
    None.

    """
    plt.plot(linedata['Date'], linedata['Cream'], linewidth=2.0)
    plt.plot(linedata['Date'], linedata['Yoghurt'], linewidth=2.0)
    plt.plot(linedata['Date'], linedata['Condensed milk'], linewidth=2.0)
    plt.plot(linedata['Date'], linedata['Butter'], linewidth=2.0)
    # Add lables, Title Legend and vlaues of X and Y axis
    plt.ylabel('Milk used in the Production(Million litres)')
    plt.xlabel('Years')
    plt.xlim(2011, 2022)
    plt.ylim(250, 600)
    plt.legend(["Cream", "Yoghurt", "Condensed milk", "Butter"])
    plt.title("Milk used in the production of milk products(2011-22)in the UK")
    # Save the Image of graph in local directiory
    plt.savefig("line1.png", dpi=300, bbox_inches='tight')
    plt.show()


def bar_plot(bardata):
    """
    The fucntion bar_plot creates Stacked Bar Graph for Tokyo 2020 Olympics
    Medal Count and The Graph is also saved in local directory

    Parameters
    ----------
    bardata : This a DataFrame to plot Stacked Bar for Tokyo 2020 Olympics
    Medal. The DataFrame is passed as parameter while calling the function.

    Returns
    -------
    None.

    """
    plt.figure()
    plt.bar(bardata['Team'], bardata['Gold'], width=0.3, label="Gold")
    plt.bar(bardata['Team'], bardata['Silver'], bottom=bardata['Gold'],
            width=0.3, label="Silver")
    plt.bar(bardata['Team'], bardata['Bronze'], bottom=list(bardata['Gold']
            + bardata['Silver']), width=0.3, label="Bronze")
    # Add lables, Title, Legend and vlaues of Y axis
    plt.title("Tokyo 2020 Olympics Medal Count")
    plt.xlabel('Countries')
    plt.ylabel("Total number of medals")
    plt.legend()
    plt.ylim(10, 120)
    # Save the Image of graph in local directiory
    plt.savefig("bar.png", dpi=300, bbox_inches='tight')
    plt.show()


def pie_plot(piedata):
    """
    The fucntion pie_plot creates Pie Chart for Revenue of the companies
    in millions of dollars in the year 2017 and 2021.
    The Graph is also saved in local directory.
    Parameters
    ----------
    piedata : This a DataFrame to plot Pie Chart for for Revenue of the
    companies in millions of dollars in the year 2017 and 2021.
    The DataFrame is passed as parameter while calling the function.

    Returns
    -------
    None.

    """
    plt.figure()

    plt.pie(piedata['Revenues($M) 2017'], autopct='%1.1f%%', labels=piedata
            ['Company Name'], pctdistance=0.7, labeldistance=1.1,
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'})
    plt.title("Revenue of the companies in millions of dollars in 2017")
    # Save the Image of graph in local directiory
    plt.savefig("Pie1.png", dpi=200, bbox_inches='tight')
    plt.show()

    plt.pie(piedata['Revenues($M) 2021'], autopct='%1.1f%%', labels=piedata
            ['Company Name'], pctdistance=0.7, labeldistance=1.1,
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'})
    plt.title("Revenue of the companies in millions of dollars in 2021")
    # Save the Image of graph in local directiory
    plt.savefig("Pie.png", dpi=200, bbox_inches='tight')
    plt.show()

# Reading the files to the dataframes


linedata = pd.read_csv('Disposals of milk.csv')
bardata = pd.read_csv('Tokyo 2020 Olympics Medal Count.csv')
piedata = pd.read_csv('Revenue.csv')

# Calling functions to create plots


line_plot(linedata)
bar_plot(bardata)
pie_plot(piedata)
