# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:47:18 2021 by Chris ten Dam.
This code has been written for an academic study titled "Driving energy and emissions: the built environment influence on car efficiency".

It couples built environment data from Statistics Netherlands and the Vitality data center.
The obtained datasets were transferred to the Snellius supercomputer to allow precise calculations of distances to city centers.
"""

import pandas as pd
import numpy as np


'''Loading the csv-datasets and removing postcodes for which essential data is missing'''
PC5_2019 = pd.read_csv('Datasets/PC5_2019.csv', index_col = 0)
PC6_2018 = pd.read_csv('Datasets/PC6_2018_Zwaartepunten_metcoordinaten.csv', usecols = ['PC6','INWONER','AFS_SUPERM','AV1_SUPERM','AV1_DAGLMD','AV1_CAFE','AV1_CAFTAR','AV1_RESTAU','AV1_BSO','AV1_KDV','AFS_OPRIT','AFS_TRNOVS','AFS_TREINS','AV1_ONDBAS','AV1_HAPRAK','Longitude','Lattitude'])
PC6_2018 = PC6_2018[PC6_2018['AFS_TREINS'] != -99997] 

'''Adding the local address density at PC5-level'''
PC5_2019 = PC5_2019[['PC5','OAD']]
PC5_2019.columns = ['PC5','OAD_PC5']
PC6_2018['PC5'] = PC6_2018['PC6'].astype(str).str[:5] #Presence letter prevents saving PC5s as integers
PC6_2018 = PC6_2018.merge(PC5_2019, left_on = 'PC5', right_on = 'PC5', how = 'left')

'''Adding the VDC-variables provided by Zhiyong Wang'''
NDVI_1000 = pd.read_csv('Datasets/VDC/NDVI_1000.csv', index_col = 0)
NDVI_1000['ndvi_avg'].replace(to_replace = -99999, value = np.nan, inplace = True)
Landuse_1000 = pd.read_csv('Datasets/VDC/Landuse_1000.csv', index_col = 0)
Crossings_1000 = pd.read_csv('Datasets/VDC/Crossings_1000.csv', index_col = 0)
Crossings_1000['f_culdesacs'] = Crossings_1000['crossing_1']/(Crossings_1000['crossing_1']+Crossings_1000['crossing_3']+Crossings_1000['crossing_4plus']) #Seems high in rural areas (one road with sideroads), can't find any classic cul-de-sac suburbs
Crossings_1000['f_4plus'] = Crossings_1000['crossing_4plus']/(Crossings_1000['crossing_1']+Crossings_1000['crossing_3']+Crossings_1000['crossing_4plus'])
Zhiyong_complete_pc6 = pd.merge(Landuse_1000, NDVI_1000, left_on = 'pc6', right_on = 'pc6', how = 'left')
Zhiyong_complete_pc6 = Zhiyong_complete_pc6.merge(Crossings_1000, left_on = 'pc6', right_on = 'pc6', how = 'left')
PC6_2018 = PC6_2018.merge(Zhiyong_complete_pc6, left_on = 'PC6', right_on = 'pc6', how = 'left')

'''Adding the municipality codes'''
PC6_per_muni = pd.read_csv('Original_Data/pc6hnr20180801_gwb-vs2.csv', sep = ';')
PC6_per_muni = PC6_per_muni.drop_duplicates(subset = 'PC6')
PC6_per_muni = PC6_per_muni[['PC6', 'Gemeente2018']]
PC6_per_muni.columns = ['PC6','Municipality']
PC6_2018 = PC6_2018.merge(PC6_per_muni, left_on = 'PC6', right_on = 'PC6', how = 'left')

'''Selecting the city center and larger city center proxies'''
PC6_2018['Destinations_1km'] = PC6_2018['AV1_SUPERM'] + PC6_2018['AV1_DAGLMD'] + PC6_2018['AV1_CAFE'] + PC6_2018['AV1_CAFTAR'] + PC6_2018['AV1_RESTAU'] + PC6_2018['AV1_BSO'] + PC6_2018['AV1_KDV'] + PC6_2018['AV1_HAPRAK'] + PC6_2018['AV1_ONDBAS'] 
Maxima = PC6_2018.groupby('Municipality', as_index = False)['Destinations_1km'].max()
Maxima.columns = ['Municipality','Maxima']
PC6_2018 = PC6_2018.merge(Maxima, left_on = 'Municipality', right_on = 'Municipality', how = 'left')
Centers = PC6_2018[PC6_2018['Destinations_1km'] == PC6_2018['Maxima']]
Centers = Centers.sort_values('landuse_idx5').drop_duplicates(subset = 'Municipality', keep='last') #If multiple PC6s have the same average number of destinations, the one with the highest landuse mix entropy index is selected
Centers = Centers[Centers['Destinations_1km'] >= 50]
Med_Centers = Centers[Centers['Destinations_1km'] >= 100]
Larger_centers = Centers[Centers['Destinations_1km'] >= 200]
Huge_centers = Centers[Centers['Destinations_1km'] >= 500] #Amsterdam, Rotterdam, The Hague, and Utrecht. Next up at 481 would be Groningen. 


'''XXXXX Only keeping postcodes that are in the MPN-datasets XXXXX''' #Cannot do this before center selection, which can in turn not be done before the landuse index addition

import os

data_directory = r"Q:\research-driving-energy"

PC62017 = pd.read_csv(data_directory + os.sep + 'MPN2017/HH_PC4_private2017.csv', usecols = ['WOONPC6'])
PC62018 = pd.read_csv(data_directory + os.sep + 'MPN2018/HH_PC4_private2018.csv', usecols = ['WOONPC6'])
PC62019 = pd.read_csv(data_directory + os.sep + 'MPN2019_csv/HH_PC4_private2019.csv', usecols = ['WOONPC6'])

PC6indata = pd.concat([PC62017, PC62018, PC62019])
PC6indata = PC6indata.sort_values(by = ['WOONPC6'])
PC6indata = PC6indata.reset_index(drop = True)
PC6indata = PC6indata[PC6indata['WOONPC6'] != '99']
PC6indata = PC6indata.drop_duplicates()

PC6_2018indata = PC6_2018[PC6_2018['PC6'].isin(PC6indata['WOONPC6'])]

AllCenters = PC6_2018[PC6_2018['Destinations_1km'] >= 50]
AllMed_Centers = PC6_2018[PC6_2018['Destinations_1km'] >= 100]
AllLarger_centers = PC6_2018[PC6_2018['Destinations_1km'] >= 200]
AllHuge_centers = PC6_2018[PC6_2018['Destinations_1km'] >= 500] #Amsterdam, Rotterdam, The Hague, and Utrecht. Next up at 481 would be Groningen. 
