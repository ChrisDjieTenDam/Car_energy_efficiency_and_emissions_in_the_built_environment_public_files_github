# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:07:37 2022 by Chris ten Dam.
This code has been written for an academic study titled "Driving energy and emissions: the built environment influence on car efficiency".

It can be used to make boxplots of energy use and CO2 emissions.

"""

import os
data_directory = r"Q:\research-driving-energy"
import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt

Explore = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/MPN3years_1entrypercar.csv', usecols = ['HH_cars', 'Car_design_MPN', 'Car_year_MPN', 'Fueltype_MPN', 'Car_kg_RDW', 'SEC_TNO', 'CO2_TNO', 'SEC_RDW', 'SEC_travelcard_real', 'SEC_Travelcard_TNO', 'Type', 'CarValid', 'Total_CO2_Travelcard_TNO','TTW_CO2_Travelcard_TNO_EFfromSTREAM','WTW_CO2_Travelcard_TNO_EFfromSTREAM'])
Explore = Explore[(Explore['CarValid'] == 1)|(Explore['Type'] == 5)] #The 3498 included cars

#To get the quantiles 
Quantile = Explore['SEC_Travelcard_TNO'].quantile(0.25)
Exploration = Explore.groupby('Fueltype_MPN')['SEC_Travelcard_TNO'].describe().reset_index()

'''XXXXX Making the boxplots of NEDC vs real-world specific energy consumption XXXXX''' 

#Creating nice cartype category labels for in the Figure
Explore['Type_string'] = Explore['Type']
Explore.loc[Explore['Type_string'] == 22,'Type_string'] = 'Midlight diesel' 
Explore.loc[Explore['Type_string'] == 23,'Type_string'] = 'Midheavy diesel' 
Explore.loc[Explore['Type_string'] == 24,'Type_string'] = 'Heavy diesel' 
Explore.loc[Explore['Type_string'] == 5,'Type_string'] = '(Hybrid) electric' 
Explore.loc[Explore['Type_string'] == 11,'Type_string'] = 'Light standard' 
Explore.loc[Explore['Type_string'] == 12,'Type_string'] = 'Midlight standard' 
Explore.loc[Explore['Type_string'] == 13,'Type_string'] = 'Midheavy standard' 
Explore.loc[Explore['Type_string'] == 14,'Type_string'] = 'Heavy standard' 

Explore.loc[Explore['Type'] == 5, 'SEC_RDW'] = np.nan #Because no reliable RDW NEDC data is available for PHEVs or BEVs

cartype_order = ['(Hybrid) electric','Midlight diesel','Midheavy diesel','Heavy diesel','Light standard','Midlight standard','Midheavy standard','Heavy standard']
color_dict = dict({'Midlight diesel':'coral','Midheavy diesel':'red','Heavy diesel': 'darkred','(Hybrid) electric':'green','Light standard':'lightskyblue','Midlight standard':'deepskyblue','Midheavy standard':'blue','Heavy standard':'navy'})
sns.set(rc={'figure.figsize':(15,5)})
sns.set_style("ticks")
sns.boxplot(data = Explore, x = 'SEC_RDW', y = 'Type_string', order = cartype_order, color = 'lightgray', fliersize = 0, whis = 0, linewidth = 0.1)#The official fuel-use in light-gray in the background
sns.boxplot(data = Explore, x = 'SEC_Travelcard_TNO', y = 'Type_string', order = cartype_order, palette = color_dict, fliersize = 5, whis = 1.5, linewidth = 2)#, hue = 'Fueltype_short', palette = 'pastel') #The real-world fuel-use
plt.xlabel('The specific energy consumption of vehicles (MJ/vkm)')
plt.ylabel('The car type category fw')
plt.xlim(0,7)
plt.savefig('Boxplot_20231101.jpg', dpi = 1000)

'''XXXXX Adding the predictions of vehicle specific energy consumption XXXXX''' 

cartype_order = ['(Hybrid) electric','Midlight diesel','Midheavy diesel','Heavy diesel','Light standard','Midlight standard','Midheavy standard','Heavy standard']
color_dict = dict({'Midlight diesel':'coral','Midheavy diesel':'red','Heavy diesel': 'darkred','(Hybrid) electric':'green','Light standard':'lightskyblue','Midlight standard':'deepskyblue','Midheavy standard':'blue','Heavy standard':'navy'})
sns.set(rc={'figure.figsize':(15,5)})#,'axes.facecolor':'white', 'figure.facecolor':'white'}) #Watch out: changes all subsequent figures
sns.set_style("ticks")
sns.boxplot(data = Explore, x = 'SEC_RDW', y = 'Type_string', order = cartype_order, color = 'lightgray', fliersize = 0, whis = 0, linewidth = 0.1)#The official fuel-use in light-gray in the background
sns.boxplot(data = Explore, x = 'SEC_Travelcard_TNO', y = 'Type_string', order = cartype_order, palette = color_dict, fliersize =  5, whis = 1.5, linewidth = 0.5)#, hue = 'Fueltype_short', palette = 'pastel') #The real-world fuel-use
plt.xlabel('The energy consumption of vehicles (MJ/vkm)')
plt.ylabel('The car type category fw')
plt.xlim(0,7)
plt.axvline(x = 1.972, color = 'Green', linestyle = '-',linewidth = 3, zorder = 0) #The student in Amsterdam if she owns at least one car 
plt.axvline(x = 2.002, color = 'Green', linestyle = '-', linewidth = 3, zorder = 0) #The student at a farm in Friesland if she owns at least one car 
plt.axvline(x = 2.186, color = 'DarkMagenta', linestyle = '--', linewidth = 3, zorder = 0) #The rich family in Amsterdam if they own at least one car 
plt.axvline(x = 2.269, color = 'DarkMagenta', linestyle = '--', linewidth = 3, zorder = 0) #The rich family at a farm in Friesland if they own at least one car
plt.savefig('Boxplot_20231020_plusscenarios.jpg', dpi = 1000)

# #Making a figure of WTW emissions vs TTW emissions. 
# ExploreCO2 = Explore[Explore['WTW_CO2_Travelcard_TNO_EFfromSTREAM'].notna()]
# ExploreCO2.loc[ExploreCO2['Fueltype_MPN'] == 5, 'Type_string'] = 'Battery electric'
# ExploreCO2.loc[ExploreCO2['Fueltype_MPN'] == 4, 'Type_string'] = 'Hybrid electric'

# cartype_order = ['Battery electric','Hybrid electric','Midlight diesel','Midheavy diesel','Heavy diesel','Light standard','Midlight standard','Midheavy standard','Heavy standard']
# color_dict = dict({'Midlight diesel':'coral','Midheavy diesel':'red','Heavy diesel': 'darkred','Battery electric': 'green','Hybrid electric':'limegreen','Light standard':'lightskyblue','Midlight standard':'deepskyblue','Midheavy standard':'blue','Heavy standard':'navy'})
# color_light = dict({'Midlight diesel':'lightsalmon','Midheavy diesel':'salmon','Heavy diesel': 'sienna','Battery electric':'forestgreen','Hybrid electric': 'palegreen','Light standard':'lightblue','Midlight standard':'skyblue','Midheavy standard':'dodgerblue','Heavy standard':'steelblue'})
# sns.set(rc={'figure.figsize':(15,5)})#,'axes.facecolor':'white', 'figure.facecolor':'white'}) #Watch out: changes all subsequent figures
# sns.set_style("ticks")
# sns.boxplot(data = ExploreCO2, x = 'WTW_CO2_Travelcard_TNO_EFfromSTREAM', y = 'Type_string', order = cartype_order, palette = color_dict, fliersize = 5, whis = 1.5, linewidth = 2)
# sns.boxplot(data = ExploreCO2, x = 'TTW_CO2_Travelcard_TNO_EFfromSTREAM', y = 'Type_string', order = cartype_order, palette = color_light, fliersize = 0, whis = 0, linewidth = 0.3)
# plt.xlabel('Well-To-Wheel versus estimated total emissions fueltypes (gCO2/vkm)')
# plt.ylabel('Fueltype')
# plt.savefig('WTWvsTTW_20231023.jpg', dpi = 1000)

'''XXXXX Making the boxplots of Well-To-Wheel versus estimated total emissions XXXXX''' 

ExploreCO2 = Explore[Explore['WTW_CO2_Travelcard_TNO_EFfromSTREAM'].notna()]

ExploreCO2['Type_string'] = Explore['Type']
ExploreCO2.loc[ExploreCO2['Type_string'] == 22,'Type_string'] = 'Midlight diesel' 
ExploreCO2.loc[ExploreCO2['Type_string'] == 23,'Type_string'] = 'Midheavy diesel' 
ExploreCO2.loc[ExploreCO2['Type_string'] == 24,'Type_string'] = 'Heavy diesel' 
ExploreCO2.loc[ExploreCO2['Type_string'] == 11,'Type_string'] = 'Light gasoline' 
ExploreCO2.loc[ExploreCO2['Type_string'] == 12,'Type_string'] = 'Midlight gasoline' 
ExploreCO2.loc[ExploreCO2['Type_string'] == 13,'Type_string'] = 'Midheavy gasoline' 
ExploreCO2.loc[ExploreCO2['Type_string'] == 14,'Type_string'] = 'Heavy gasoline' 
ExploreCO2.loc[ExploreCO2['Fueltype_MPN'] == 5, 'Type_string'] = 'Battery electric'
ExploreCO2.loc[ExploreCO2['Fueltype_MPN'] == 4, 'Type_string'] = 'Gasoline hybrid electric'

cartype_order = ['Battery electric','Midlight diesel','Midheavy diesel','Heavy diesel','Light gasoline','Midlight gasoline','Midheavy gasoline','Heavy gasoline','Gasoline hybrid electric']
color_dict = dict({'Midlight diesel':'coral','Midheavy diesel':'red','Heavy diesel': 'darkred','Battery electric': 'green','Gasoline hybrid electric':'limegreen','Light gasoline':'lightskyblue','Midlight gasoline':'deepskyblue','Midheavy gasoline':'blue','Heavy gasoline':'navy'})
color_light = dict({'Midlight diesel':'lightsalmon','Midheavy diesel':'salmon','Heavy diesel': 'sienna','Battery electric':'forestgreen','Gasoline hybrid electric': 'palegreen','Light gasoline':'lightblue','Midlight gasoline':'skyblue','Midheavy gasoline':'dodgerblue','Heavy gasoline':'steelblue'})
sns.set(rc={'figure.figsize':(15,5)})
sns.set_style("ticks")
sns.boxplot(data = ExploreCO2, x = 'Total_CO2_Travelcard_TNO', y = 'Type_string', order = cartype_order, palette = color_light, fliersize = 0, whis = 0, linewidth = 0.3)
sns.boxplot(data = ExploreCO2, x = 'WTW_CO2_Travelcard_TNO_EFfromSTREAM', y = 'Type_string', order = cartype_order, palette = color_dict, fliersize = 5, whis = 1.5, linewidth = 2)
plt.xlabel('For context: Well-To-Wheel versus estimated total (faded colors) emissions of battery electric, diesel, and gasoline vehicles (gCO2eq/vkm)')
plt.ylabel('Fueltype')
plt.xlim(0,700)
plt.savefig('WTWvsTotal_20231101.jpg', dpi = 1000)
