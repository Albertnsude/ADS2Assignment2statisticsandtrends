# Importing Libraries for the purpose of this analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

"""
creating a def function to read in the world bank climate datasets and returns  two 
dataframes:one with years as columns, the other with nations

"""

def read_world_climate_data(filename):
    
    """
    A function that reads in world bank data on climate change with various 
    indicators from and return both the original and transposed version of
    the dataset
    
    Args:
        filename: the name of the world bank data that will be read for analysis 
        and manupulation
        
            
    Returns: 
        The dataset as df_world_data_years(years as column) and df_world_data_countries as its transposed version
    """  
    # Read the World Bank Climate data into a dataframe
    df_world_climate = pd.read_csv('world_climate_data.csv', skiprows=4)

    # Transpose the dataframe and set the country name as the index
    df_world_data_years = df_world_climate.drop(['Country Code', 'Indicator Code'], axis=1)
    df_world_data_countries = df_world_climate.drop(['Country Name', 'Country Code', 'Indicator Code'], axis=1) \
        .set_index(df_world_climate['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    df_world_data_countries = df_world_data_countries.set_index('Year').dropna(axis=1)
    

    return df_world_data_years, df_world_data_countries

print(read_world_climate_data)

# Reading the function that will produce two DataFrame(years as column and countries as column)
df_world_data_years, df_world_data_countries = read_world_climate_data('world_climate_data.csv')

"""
Further analysis will be carried out on the world bank climate change data
for some indicators and 7 countries 
 
"""
# Indicators and countries of choice will be selected and some statistical variations will be investigated for this purpose of the analysis
indicators = df_world_data_years[df_world_data_years['Indicator Name'].isin(["Urban population", "CO2 emissions (kt)", "Electricity production from nuclear sources (% of total)", "Electricity production from oil sources (% of total)", "Renewable energy consumption (% of total final energy consumption)"])]

countries = ['United States', 'China', 'India', 'United Kingdom', 'Russian Federation', 'France', 'Canada']
selected_countries = indicators[indicators['Country Name'].isin(countries)]
selected_countries = selected_countries.dropna(axis=1)
selected_countries = selected_countries.reset_index(drop=True)
selected_countries



"""
Investigating statistical properties of the selected indicators 
for the 7 countries under our study

"""

# describe function will be applied to this selected years '1990', '2000', '2010', '2014'
stats_desc = selected_countries.groupby(["Country Name","Indicator Name"]) \
[['1990', '2000', '2010', '2014']].describe()

print(stats_desc)

"""
Executing the summary statistics of 
mean,min,max,median and standard deviation of the indicators of the 7 countries
for years in study

"""
# Summary statistics of the indicators will be applied for the countries and years
summary_stats_others = selected_countries.groupby(['Country Name', 'Indicator Name'])
for name, group in summary_stats_others:
    print(name)
    print('Mean:', group.mean()['1990':'2014'])
    print('Min:', group.min()['1990':'2014'])
    print('Max:', group.max()['1990':'2014'])
    print('Median:', group.median()['1990':'2014'])
    print('Standard deviation:', group.std()['1990':'2014'])

"""
The statistical properties of each selected indicators will be executed
and comparison amongst the 7 countries and the indicators will be performed
for a more critical summary statistics by creating new dataframe for each indicators for the analysis

"""

# Creating a dataFrame for Urban population for statistics analysis and plotting.
Urban_pop = selected_countries[selected_countries["Indicator Name"] == "Urban population"]
Urban_pop = Urban_pop.set_index('Country Name', drop=True)
Urban_pop= Urban_pop.transpose().drop('Indicator Name')
Urban_pop[countries] = Urban_pop[countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(Urban_pop)

# Statistic summary for urban population
print(Urban_pop.describe())
print(Urban_pop.mean()) 
print(Urban_pop.median()) 
print(Urban_pop.std())
print('Skewness:', Urban_pop.skew())

   
#Creating a dataFrame for CO2 emissions (kt) for statistics analysis and plotting.
CO2_emi = selected_countries[selected_countries["Indicator Name"] == "CO2 emissions (kt)"]
CO2_emi = CO2_emi.set_index('Country Name', drop=True)
CO2_emi = CO2_emi.transpose().drop('Indicator Name')
CO2_emi [countries] = CO2_emi [countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(CO2_emi)

# Statistic summary for CO2 emissions (kt)
print(CO2_emi.describe())
print(CO2_emi.mean())
print(CO2_emi.median())
print(CO2_emi.std())
print('Skewness:', CO2_emi.skew())

# Creating a dataFrame for Electricity production from nuclear sources (% of total)for statistics analysis and plotting.
Elect_prod_nuc = selected_countries[selected_countries["Indicator Name"] == \
                                        "Electricity production from nuclear sources (% of total)"]
Elect_prod_nuc = Elect_prod_nuc.set_index('Country Name', drop=True)
Elect_prod_nuc = Elect_prod_nuc.transpose().drop('Indicator Name')
Elect_prod_nuc[countries] = Elect_prod_nuc[countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(Elect_prod_nuc)

# Statistic summary for Electricity production from nuclear sources (% of total)
print(Elect_prod_nuc.describe())
print(Elect_prod_nuc.mean())
print(Elect_prod_nuc.median())
print(Elect_prod_nuc.std())
print('Skewness:', Elect_prod_nuc.skew())

# Creating a dataFrame for Electricity production from oil sources (% of total) for statistics analysis and plotting.
Elect_prod_oil = selected_countries[selected_countries["Indicator Name"] == \
                                        "Electricity production from oil sources (% of total)"]
Elect_prod_oil = Elect_prod_oil.set_index('Country Name', drop=True)
Elect_prod_oil = Elect_prod_oil.transpose().drop('Indicator Name')
Elect_prod_oil[countries] = Elect_prod_oil[countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(Elect_prod_oil)

# Statistic summary for Electricity production from oil sources (% of total)
print(Elect_prod_oil.describe())
print(Elect_prod_oil.mean())
print(Elect_prod_oil.median())
print(Elect_prod_oil.std())
print('Skewness:', Elect_prod_oil.skew())

# Creating a dataFrame for Renewable energy consumption for statistics analysis and plotting.
Renew_Energy = selected_countries[selected_countries["Indicator Name"] \
                                       == "Renewable energy consumption (% of total final energy consumption)"]
Renew_Energy = Renew_Energy.set_index('Country Name', drop=True)
Renew_Energy = Renew_Energy.transpose().drop('Indicator Name')
Renew_Energy[countries] = Renew_Energy[countries].apply(pd.to_numeric, errors='coerce', axis=1)
print(Renew_Energy)

# Statistic summary for Renewable energy consumption
print(Renew_Energy.describe())
print(Renew_Energy.mean())
print(Renew_Energy.median())
print(Renew_Energy.std())
print('Skewness:', Renew_Energy.skew())

"""
Plotting line plots of CO2 emissions and Urban Population 
to show the trends over years for 7 countries

"""

# The line plot for CO2 emissions
plt.figure(figsize=(10,6))
plt.style.use('default')
CO2_emi.plot()
plt.title('CO2 emmissions of 7 countries')
plt.xlabel('Year')
plt.ylabel('"CO2 emissions (kt)')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Line plot CO2 emmisions.png', dpi=300)
plt.show()

# The Line plot for Urban Population
plt.figure(figsize=(10,6))
plt.style.use('default')
Urban_pop.plot()
plt.title('Urban Population of 7 countries)')
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Line plot Urban Population.png', dpi=300)
plt.show()


"""
Plotting a grouped bar chart to show the Electricity production from oil sources (% of total) for 7 countries

"""

plt.style.use('seaborn')

# Create a new column with the decade for each year
CO2_emi['Decade'] = CO2_emi.index.map(lambda x: str(x)[:3] + '0s')

# Group by decade and sum the CO2 emission (kt) for each country
CO2_decade = CO2_emi.groupby('Decade').sum()

colors = {'United States': 'orange', 'Canada': 'brown', 'United Kingdom': 'black', 'France': 'green',
          'Russia Federation': 'grey', 'China': 'blue', 'India': 'yellow'}

# Create a new column with the decade for each year
Elect_prod_oil['Decade'] = Elect_prod_oil.index.map(lambda x: str(x)[:3] + '0s')

# Group by decade and sum the Electricity production from oil sources for each country
Elect_prod_oil = Elect_prod_oil.groupby('Decade').sum()

# Plot the data as a grouped bar chart for Electricity production from oil sources for the countries
Elect_prod_oil.plot(kind='bar', color=[colors.get(c, 'red') for c in Elect_prod_oil.columns])
plt.title('Electricity Production (Oil Sources by 7 Countries)')
plt.xlabel('Decade')
plt.ylabel('Electricity Production (Oil Sources) (% of total)')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('grouped barchat Electricity production from Oil Sources.png', dpi=300)
plt.show()

"""
Plotting a scatter plot to show relationship for Urban poulation and Renewable energy
consumption in United States
"""
plt.style.use('ggplot')
plt.scatter(Urban_pop['United States'], Renew_Energy['United States'])
plt.title('R/ship btw Urban Pop & Ren. Energy ons. in United States')
plt.xlabel('Urban population')
plt.ylabel('Renewable energy consumption')
plt.savefig('scatter plot United States.png', dpi=300)
plt.show()

"""
Plotting a scatter plot to show relationship for CO2 emmission and Renewable energy
consumption in China
"""
plt.style.use('ggplot')
plt.scatter(Renew_Energy['China'], CO2_emi['China'])
plt.title('R/ship btw CO2 emmision & Ren. energy cons. in China')
plt.xlabel('Renewable energy consumption')
plt.ylabel('CO2 emmission')
plt.savefig('scatter plot China.png', dpi=300)
plt.show()

"""
Plotting a heatmap to show the correlation that displays data using different shades or colors to represent different variable across India

"""
# crating a DataFrame to from the dictionaries containing the different indicators
India_country = pd.DataFrame({'CO2 emi (kt)': CO2_emi['India'], 'Urban pop': Urban_pop['India'], \
                            'Elec Prod Oil': Elect_prod_oil['India'], \
                            'Elec Prod Nuclear': Elect_prod_nuc['India'], \
                            'Ren Enery': Renew_Energy['India']}, \
                             index=['1990', '1995', '2000', '2005', '2010', '2015', '2019'])

India_country.corr()

#plotting the heatmap
plt.figure(figsize=(8,5))
sns.heatmap(India_country.corr(), annot=True, cmap='Greens')
plt.title('Correlation heatmap for india')
plt.savefig('Heatmap India.png', dpi=300)
plt.show()
