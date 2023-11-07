# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:47:03 2022 by Chris ten Dam.
This code has been written for an academic study titled "Driving energy and emissions: the built environment influence on car efficiency".

It is meant to be run on the Snellius supercomputer and supplements the built environment data with computed distances to city centers.
"""

import pandas as pd
import numpy as np
from geopy import distance

PC6_2018indata = pd.read_csv('/gpfs/home3/dam00037/PC62018/PC6_2018indata.csv')
Centers = pd.read_csv('/gpfs/home3/dam00037/PC62018/Centers.csv')
Med_Centers = pd.read_csv('/gpfs/home3/dam00037/PC62018/Med_Centers.csv')
Larger_centers = pd.read_csv('/gpfs/home3/dam00037/PC62018/Larger_centers.csv')
Huge_centers = pd.read_csv('/gpfs/home3/dam00037/PC62018/Huge_centers.csv')
AllCenters = pd.read_csv('/gpfs/home3/dam00037/PC62018/AllCenters.csv')
AllMed_Centers = pd.read_csv('/gpfs/home3/dam00037/PC62018/AllMed_Centers.csv')
AllLarger_centers = pd.read_csv('/gpfs/home3/dam00037/PC62018/AllLarger_centers.csv')
AllHuge_centers = pd.read_csv('/gpfs/home3/dam00037/PC62018/AllHuge_centers.csv')

from geopy import distance

for postcode in PC6_2018indata.index:
    distances = []
    for center in Huge_centers.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], Huge_centers.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_hugecenter'] = min_dist

for postcode in PC6_2018indata.index:
    distances = []
    for center in Larger_centers.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], Larger_centers.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_largercenter'] = min_dist

for postcode in PC6_2018indata.index:
    distances = []
    for center in Med_Centers.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], Med_Centers.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_mediumcenter'] = min_dist

for postcode in PC6_2018indata.index:
    distances = []
    for center in Centers.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], Centers.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_center'] = min_dist

CSVready = PC6_2018indata[['PC6','OAD_PC5','AFS_SUPERM','AFS_OPRIT','AFS_TRNOVS','AFS_TREINS','km_center','km_mediumcenter','km_largercenter','km_hugecenter','ndvi_avg','landuse_idx5','f_4plus']]
CSVready.to_csv('/gpfs/home3/dam00037/PC6_2018_bewerkt_standardcenters.csv')

'''Computation distances to all PC6 with minimum level of destinations (high computing power required!)'''

for postcode in PC6_2018indata.index:
    distances = []
    for center in AllHuge_centers.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], AllHuge_centers.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_allhugecenters'] = min_dist
CSVready = PC6_2018indata[['PC6','OAD_PC5','AFS_SUPERM','AFS_OPRIT','AFS_TRNOVS','AFS_TREINS','km_center','km_mediumcenter','km_largercenter','km_hugecenter', 'km_allhugecenters','ndvi_avg','landuse_idx5','f_4plus']]
CSVready.to_csv('/gpfs/home3/dam00037/PC6_2018_bewerkt_plusallhugecenters.csv')

for postcode in PC6_2018indata.index:
    distances = []
    for center in AllLarger_centers.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], AllLarger_centers.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_alllargercenters'] = min_dist
CSVready = PC6_2018indata[['PC6','OAD_PC5','AFS_SUPERM','AFS_OPRIT','AFS_TRNOVS','AFS_TREINS','km_center','km_mediumcenter','km_largercenter','km_hugecenter', 'km_alllargercenters', 'km_allhugecenters','ndvi_avg','landuse_idx5','f_4plus']]
CSVready.to_csv('/gpfs/home3/dam00037/PC6_2018_bewerkt_plusalllargerhugecenters.csv')

for postcode in PC6_2018indata.index:
    distances = []
    for center in AllMed_Centers.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], AllMed_Centers.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_allmediumcenters'] = min_dist
CSVready = PC6_2018indata[['PC6','OAD_PC5','AFS_SUPERM','AFS_OPRIT','AFS_TRNOVS','AFS_TREINS','km_center','km_mediumcenter','km_largercenter','km_hugecenter', 'km_allhugecenters', 'km_alllargercenters', 'km_allmediumcenters','ndvi_avg','landuse_idx5','f_4plus']]
CSVready.to_csv('/gpfs/home3/dam00037/PC6_2018_bewerkt_plusallmediumlargerhugecenters.csv')

for postcode in PC6_2018indata.index:
    distances = []
    for center in AllCenters.index:
        distances.append(distance.distance(PC6_2018indata.loc[postcode, ['Lattitude','Longitude']], AllCenters.loc[center,['Lattitude','Longitude']]).km)
    min_dist = min(distances)
    PC6_2018indata.loc[postcode, 'km_allcenters'] = min_dist
CSVready = PC6_2018indata[['PC6','OAD_PC5','AFS_SUPERM','AFS_OPRIT','AFS_TRNOVS','AFS_TREINS','km_center','km_mediumcenter','km_largercenter','km_hugecenter', 'km_allhugecenters', 'km_alllargercenters', 'km_allmediumcenters', 'km_allcenters','ndvi_avg','landuse_idx5','f_4plus']]
CSVready.to_csv('/gpfs/home3/dam00037/PC6_2018_bewerkt_plusallsmallmediumlargerhugecenters.csv')
