# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:53:01 2023

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



"""
generating a def function to read in the world bank climate datasets and returns  two 
dataframes:one with years as columns, the other with nations
"""

def read_data (filename, **others):
    """
    A function that reads in world climate data change alongside various indicators from 
    the worldbank climate database and returns both the original and transposed version of
    the dataset
    
    Args:
        filename: the name of the world climate data that will be read for the purpose of analysis 
        and manupulation
        
        **others: other arguments to pass into the functions as need be
            
    Returns: 
        The original csv dataset format as obtained from the world bank report and its transposed version
    """        

    # The csv world bank climate data file is read into dataframe for analysis purpose with years as columns
    world_climate_data = pd.read_csv(filename, skiprows=4) 
    
    # Transposing and cleaning the dataset such that the country names are the columns
    world_climate_data2 = pd.DataFrame.transpose(world_climate_data)
    world_climate_data2=world_climate_data2.drop(['Country Name','Country Code','Indicator Code'])
    world_climate_data2.columns=world_climate_data2.iloc[0]
    world_climate_data2=world_climate_data2.iloc[1:]
    
    return world_climate_data, world_climate_data2

# Reading in our datasets into dataframe for analysis purpose

world_climate_data, world_climate_data2 = read_data('world_bank_climate_data.csv')
print(world_climate_data)

# 4 indicators of choice will be selected for this purpose of the analysis

indicators=world_climate_data[world_climate_data['Indicator Name'].isin(['Arable land (% of land area)','Urban population',
                                                  'CO2 emissions (kt)','Electric power consumption (kWh per capita)'])]

print(indicators.head())

# Transposing the world bank climate data

"""
The world bank data looks so bogus, complex and uneasy to work with, 
so various data functioning exercises will be executed in order to make the data accessible and easier to analyze
"""

world_climate_data=pd.DataFrame.transpose(indicators) 
print(world_climate_data)

# permuting the country name as columns.
world_climate_data.columns=world_climate_data.iloc[0]
print(world_climate_data)

# dropping rows not important from the world bank climate data
world_climate_data=world_climate_data.drop(['Country Name','Country Code','Indicator Code'])
print(world_climate_data)

"""
For this analysis, 7 countries of choice across various continents
will be selected for proper understanding and depiction of the climate change
"""
# Choosing countries for purpose of analysis
countries= world_climate_data[['China', 'Nigeria', 'India','Philippines','United States', 'Germany','Brazil']]

print(countries)

# Checking for missing values because We have lots of missing values, some columns having 50% of its values missing

countries.isnull().mean() 
print(countries.isnull().mean())

# dropping all missing values from our world climate bank datasets

countries.dropna(inplace=True)
countries.head()
countries.index
print(countries)

"""
future accessibilty and ease of data analytical datasets that combine all 
countries on each indicator will be created in five years increments from 1990 to 2010
"""
# Creating a dataframe for all selected countries on Urban population
urban_pop=countries.iloc[[1,6,11,16,21],[0,4,8,12,16,20,24]]

# transforming from dataframe data type to a numeric format
urban_pop=urban_pop.apply(pd.to_numeric) 

# transforming from index values to a numeric format
urban_pop.index=pd.to_numeric(urban_pop.index) 
print(urban_pop)

# creating a dataframe for all selected countries on c02 emission
co2= countries.iloc[[1,6,11,16,21],[1,5,9,13,17,21,25]] 

# transforming from data type to a numeric format
co2=co2.apply(pd.to_numeric) 

# transforming from index values to a numeric format
co2.index=pd.to_numeric(co2.index) 
print(co2)

# creating a dataframe for all selected countries on electric consumption 
electric=countries.iloc[[1,6,11,16,21],[2,6,10,14,18,22,26]]

# transforming from data type to a numeric format
electric=electric.apply(pd.to_numeric) 

# transforming from index values to a numeric format
co2.index=pd.to_numeric(co2.index) 
print(electric)

# creating a dataframe for all selected countries on Arable land
arable=countries.iloc[[1,6,11,16,21],[3,7,11,15,19,23,27]]

# transforming from data type to a numeric format
arable=arable.apply(pd.to_numeric)

# transforming from index values to a numeric format
arable.index=pd.to_numeric(arable.index) 
print(arable.head())

"""
Using Statistical overview of the urban population across the 7 selected countries usin describe function and
applying some Statistical functions for the 4 selected indicators across the 7 selected countries will suffice now."""

#statistical function for urban population
print(urban_pop.describe())

# checking the mean urban population
print(urban_pop.mean()) 

 # checking the median urban population
print(urban_pop.median())

#Statistical function for c02 emission
print(co2.describe())

# checking the mean co2
print(co2.mean()) 

# checking the median co2
print(co2.median())

# checking the co2 standard deviation
print(co2.std()) 

 # checking the urban population standard deviation
print(urban_pop.std())

#Statistical function for electric consumption 
print(electric.describe())

# checking the mean co2
print(electric.mean()) 

 # checking the median co2
print(electric.median())

 # checking the co2 standard deviation
print(electric.std())

#Statistical function for Arable Land
print(arable.describe())

 # checking the mean for arable land
print(arable.mean())

# checking the median arable land
print(arable.median()) 

# checking the arable land standard deviation
print(arable.std()) 

print(plt.style.available)

# Plotting a grouped bar of CO2 Emission for the 7 Countries

"""
Plotting of a grouped bar of CO2 emission for the 7 countries  in 5 years increments
from the year 1990 to 2010 
"""
plt.style.use('default')
co2.T.plot(kind='bar')
plt.title(' nations co2 emission in 5 years increments')
plt.xlabel('Countries')
plt.ylabel('Co2 emission (kt)')
plt.show()

# Plotting a line plot of the Urban Population Trend for the 7 countries

plt.figure(figsize=(10,6))
plt.style.use('default')
urban_pop.plot()
plt.title('Urban Population Trend for The 7 countries')
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.xticks([1990,1995,2000,2005,2010])
plt.show()

# plotting a scatter plot to show relationship for CO2 emissions and urban population growth in china

"""
Plotting of a scatter plot to show relationship for CO2 emissions and urban population growth in china in 5 years increments
from the year 1990 to 2010 
"""
plt.style.use('ggplot')
plt.scatter(co2['China'], urban_pop['China'])
plt.title('Relationship between Co2 emission (kt) and Urban population in China')
plt.xlabel('Co2 emission (kt)')
plt.ylabel('Urban population')
plt.show()

