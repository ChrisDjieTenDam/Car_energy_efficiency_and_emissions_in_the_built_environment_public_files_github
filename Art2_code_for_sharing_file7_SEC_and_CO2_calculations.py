# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:36:28 2022 by Chris ten Dam.
This code has been written for an academic study titled "Car energy efficiency and emissions in the built environment".

It combines the MPN data from 2017, 2018, and 2019.
Next, it computes the CO2 emissions and energy use of each vehicle using TNO (de Ruiter et al. 2021) and CE Delft (STREAM 2023) research.
Finally, it defines the fuel- and weight-based cartypes and marks cars of which important data is missing. 
"""

import os
data_directory = r"Q:\research-driving-energy"

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#Get the files. These are the cleaned files per MPN-year with public and private data from a number of sources (CBS, RDW, etc.) combined but with the identifiers (postcodes, license plates, etc.) removed
MPN2017combi = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/MPN2017_combinedfile_1entrypercar.csv', index_col = 0) 
MPN2018combi = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/MPN2018_combinedfile_1entrypercar.csv', index_col = 0)
MPN2019combi = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/MPN2019_combinedfile_1entrypercar.csv', index_col = 0)

##Preventing duplicate entries such that a car sold in 2018 will not show up in the 2019 dataset and so on.
MPN2018combi = MPN2018combi[~MPN2018combi['HHID'].isin(MPN2019combi['HHID'])] 
MPN2017combi = MPN2017combi[~MPN2017combi['HHID'].isin(MPN2019combi['HHID'])] 
MPN2017combi = MPN2017combi[~MPN2017combi['HHID'].isin(MPN2018combi['HHID'])]

MPN3years = pd.concat([MPN2017combi, MPN2018combi, MPN2019combi]) #Combining the three waves to avoid underfitting #5440 if removing HHs present in earlier years first
MPN3years = MPN3years.reset_index(drop = True) #Because duplicate index values, which cause problems later on 


'''XXXXX The MJ/vkm and gCO2/vkm calculations (TNO models) XXXXX'''

#First setting up the calculations in accordance with equation de Ruiter et al. (2021): CO2 = a*M + b(t) + c1*P/M + c2
#Then executing equations and correcting for H2 and LPG vehicles (the latter were previously treated as petrol vehicles)
#Then imputing for missing data from HEVs, BEVs, and PHEVs (all of which are retained, regardless of their weight)
#Then making computations of MJ/vkm and CO2 for HEVs, BEVs, and PHEVs and converting to CO2 (as based on TNO research, see also the CarData file in which data are coupled for each battery or plug-in hybrid carmodel)
#Then ensuring all energy values (including from RDW) are in MJ/vkm
#Finally combining the data and setting the cartype and carvalid indicators

vkm = MPN3years.copy() 
vkm = vkm[['CARID','Car_year_MPN','Fueltype_MPN','Car_kg_RDW','Nettmaxpower_RDW','Fuel_RDW_combined','CO2_RDW_combined','vkms_elec_TNO','MJ/vkm_EVs_TNO','MJ/vkm_fuel_PHEV_TNO']] #Car_kg_RDW = Massa rijklaar = Massa ledig voertuig + 100 except for 6 cases with minor deviations
vkm['P/M'] = vkm['Nettmaxpower_RDW']/(vkm['Car_kg_RDW']/1000) #Engine power divided by vehicle mass in tonnes, In accordance with instructions in de Ruiter et al. (2021)
vkm['Ma'] = 0 #Note: Car kg empty = Massa ledig voertuig from RDW. Massa rijklaar from RDW = Massa ledig voertuig + 100 kg except in a few cases. However, Massa ledig voertuig is defined more often and the MPN also stores Massa ledig voertuig (GEWLEEG)
vkm['b(t)'] = 0 
vkm['correction'] = 0 
vkm['Age_in_2020'] = 2020 - vkm['Car_year_MPN'] #To avoid issues. Is only to couple the building years correctly.

#Importing the fuel-type specific age-correction factors b(t)
bt = pd.read_excel('bt_TNO.xlsx')
bt['bt_age_in_2020'] = 2020 - bt['Build year\n𝒃(𝒕)']
bt.columns = ['Build year','Petrol','Diesel','Petrol Hybrid','Petrol PHEV','Petrol PHEV elec','Diesel PHEV','Diesel PHEV elec','bt_age_in_2020'] #To avoid issues
bt = bt.sort_values(by = 'bt_age_in_2020')
bt.index = bt['bt_age_in_2020']

#Use the Table for ages up to 15 and then extrapolate based on latest values (based on cars with ages 14 and 15, built in 2005 and 2006) #You cannot check the extrapolation process with Praktijkverbruik data
bt13 = bt[bt['bt_age_in_2020'] > 13]
SLR = LinearRegression()
SLR.fit(bt13['bt_age_in_2020'].values.reshape(-1,1), bt13['Petrol'])
bt_petrol_extrapolated = SLR.predict(np.array(range(118)).reshape(-1,1))
SLR.fit(bt13['bt_age_in_2020'].values.reshape(-1,1), bt13['Diesel'])
bt_diesel_extrapolated = SLR.predict(np.array(range(118)).reshape(-1,1))
bt_petrol = bt['Petrol'].squeeze()
bt_diesel = bt['Diesel'].squeeze()
bt_hybrid = bt['Petrol Hybrid'].squeeze()

#Determination a*M, b(t), and correction term "c1*P/M + c2" for gasoline vehicles
vkm.loc[vkm['Fueltype_MPN'] == 1, 'Ma'] = vkm['Car_kg_RDW']*0.0812
for year in range(16):
    vkm.loc[(vkm['Fueltype_MPN'] == 1) & (vkm['Age_in_2020'] == year), 'b(t)'] = bt_petrol[year]
for year in range(16,100):
    vkm.loc[(vkm['Fueltype_MPN'] == 1) & (vkm['Age_in_2020'] == year), 'b(t)'] = bt_petrol_extrapolated[year] 
vkm.loc[(vkm['Fueltype_MPN'] == 1) & (vkm['P/M'] >= 90) & (vkm['P/M'] <= 220), 'correction'] = 0.389*vkm['P/M'] - 28.8
vkm.loc[(vkm['Fueltype_MPN'] == 1) & (vkm['P/M'] > 220), 'correction'] = 56.7 #Decided to group the thresshold of 220 under the middle-power group (the line above), doesn't really matter as the border is matched up: 0.389*220-28.8 = 56.8

#Determination a*M, b(t), and correction term "c1*P/M + c2" for diesel vehicles
vkm.loc[(vkm['Fueltype_MPN'] == 2), 'Ma'] = vkm['Car_kg_RDW']*0.1194
for year in range(16):
    vkm.loc[(vkm['Fueltype_MPN'] == 2) & (vkm['Age_in_2020'] == year), 'b(t)'] = bt_diesel[year]
for year in range(16,100):
    vkm.loc[(vkm['Fueltype_MPN'] == 2) & (vkm['Age_in_2020'] == year), 'b(t)'] = bt_diesel_extrapolated[year] 
vkm.loc[(vkm['Fueltype_MPN'] == 2) & (vkm['P/M'] <= 30), 'correction'] = 39.2
vkm.loc[(vkm['Fueltype_MPN'] == 2) & (vkm['P/M'] > 30) & (vkm['P/M'] <= 55), 'correction'] = -1.51*vkm['P/M'] + 84.4 #Decided to group the thresshold of 30 under the lowest-power group (the line above), doesn't really matter as the border is matched up: -1.51*30+84.4 = 39.1
# vkm.loc[(vkm['Fueltype_MPN'] == 2) & (vkm['P/M'] >= 100), 'correction'] = 0 #See mail van Gijlswijk 20220721

#Determination a*M, b(t), and correction term "c1*P/M + c2" for petrol-electric hybrid vehicles
vkm.loc[(vkm['Fueltype_MPN'] == 4), 'Ma'] = vkm['Car_kg_RDW']*0.0646
for year in range(15):
    vkm.loc[(vkm['Fueltype_MPN'] == 4) & (vkm['Age_in_2020'] == year), 'b(t)'] = bt_hybrid[year]
for year in range(15,50): #Weird values b(t): no clear trend, so using mean
    vkm.loc[(vkm['Fueltype_MPN'] == 4) & (vkm['Age_in_2020'] == year), 'b(t)'] = bt_hybrid.mean() 

#The application fo the equation (in gCO2/vkm) and the correction for the H2-vehicle
vkm['CO2_TNO'] = vkm['Ma'] + vkm['b(t)'] + vkm['correction']
vkm.loc[vkm['Fueltype_MPN'] == 7, 'CO2_TNO'] = 0  #TTW Emissions of hydrogen are always zero: they emit water


'''XXXXX Imputations, including data on EVs from other TNO reports, and converting CO2 into SEC XXXXX '''

#Conversion factors for adding TNO data
MJ_per_liter_petrol = 31.350 #0.75 kg/L E10 gasoline *41.8 MJ/kg. Source: STREAM 2023 (page 94) 
MJ_per_liter_diesel = 35.952 #0.84 kg/L B7 diesel *42.8 MJ/kg. Source: STREAM 2023 (page 94) 
MJ_per_liter_LPG = 24.408 #0.54 kg/L * 45.2 MJ/kg LPG. Source: STREAM 2023 (page 94). 
MJperkWh_over100 = 3.6/100

gCO2_per_MJ_petrol = 2370/MJ_per_liter_petrol #2370 gCO2/liter gasoline according to de Ruiter et al. (2021). Need to use this value as it is the one TNO (Ruiter et al.) used to convert energy into CO2 emissions. 
gCO2_per_MJ_diesel = 2650/MJ_per_liter_diesel #2650 gCO2/liter diesel according to de Ruiter et al. (2021) 

#Imputation of values for petrol hybrid vehicles. All HEVS are included in the regression analyses so if year and/or mass unknown: take average
for year in range(50):
    select = vkm[(vkm['Fueltype_MPN'] == 4) & (vkm['Age_in_2020'] == year)  & (vkm['CO2_TNO'] != 0)] 
    vkm.loc[(vkm['Fueltype_MPN'] == 4) & (vkm['Age_in_2020'] == year) & (vkm['CO2_TNO'].isna()), 'CO2_TNO'] = select['CO2_TNO'].mean()
    vkm.loc[(vkm['Fueltype_MPN'] == 4) & (vkm['Age_in_2020'] == year) & (vkm['CO2_TNO'] == 0), 'CO2_TNO'] = select['CO2_TNO'].mean() #Just to be sure
select = vkm[(vkm['Fueltype_MPN'] == 4) & (vkm['CO2_TNO'] != 0)] 
vkm.loc[(vkm['Fueltype_MPN'] == 4) & (vkm['CO2_TNO'].isna()), 'CO2_TNO'] = select['CO2_TNO'].mean() 
vkm.loc[(vkm['Fueltype_MPN'] == 4) & (vkm['CO2_TNO'] == 0), 'CO2_TNO'] = select['CO2_TNO'].mean() #Just to be sure

#No estimation for BEVs: kWh/100km taken directly from Ruiter et al. 2021 data (pg22)
select =  vkm[vkm['Fueltype_MPN'] == 5]
vkm.loc[(vkm['Fueltype_MPN'] == 5) & (vkm['MJ/vkm_EVs_TNO'].isna()), 'MJ/vkm_EVs_TNO'] = select['MJ/vkm_EVs_TNO'].mean() #For the one Fiat that is not in the TNO data
vkm.loc[(vkm['Fueltype_MPN'] == 5), 'CO2_TNO'] = 0 #TTW CO2 emissions for elec are zero

#Imputation MJ/vkm and computation gCO2/vkm for PHEVs based on fraction electric kilometers travelled
select =  vkm[vkm['Fueltype_MPN'] == 15]
vkm.loc[(vkm['Fueltype_MPN'] == 15) & (vkm['MJ/vkm_EVs_TNO'].isna()), 'MJ/vkm_EVs_TNO'] = select['MJ/vkm_EVs_TNO'].mean()
vkm.loc[(vkm['Fueltype_MPN'] == 15) & (vkm['vkms_elec_TNO'].isna()), 'vkms_elec_TNO'] = select['vkms_elec_TNO'].mean()
vkm.loc[(vkm['Fueltype_MPN'] == 15) & (vkm['MJ/vkm_fuel_PHEV_TNO'].isna()), 'MJ/vkm_fuel_PHEV_TNO'] = select['MJ/vkm_fuel_PHEV_TNO'].mean()
vkm.loc[(vkm['Fueltype_MPN'] == 15), 'CO2_TNO'] = vkm['MJ/vkm_fuel_PHEV_TNO']*(1-vkm['vkms_elec_TNO'])*gCO2_per_MJ_petrol #Fuel_per_non_electric km*Non_electric_kms*gCO2_per_MJ #gasoline PHEV uses gasoline
select =  vkm[vkm['Fueltype_MPN'] == 25]
vkm.loc[(vkm['Fueltype_MPN'] == 25) & (vkm['MJ/vkm_EVs_TNO'].isna()), 'MJ/vkm_EVs_TNO'] = select['MJ/vkm_EVs_TNO'].mean()
vkm.loc[(vkm['Fueltype_MPN'] == 25) & (vkm['vkms_elec_TNO'].isna()), 'vkms_elec_TNO'] = select['vkms_elec_TNO'].mean()
vkm.loc[(vkm['Fueltype_MPN'] == 25) & (vkm['MJ/vkm_fuel_PHEV_TNO'].isna()), 'MJ/vkm_fuel_PHEV_TNO'] = select['MJ/vkm_fuel_PHEV_TNO'].mean()
vkm.loc[(vkm['Fueltype_MPN'] == 25), 'CO2_TNO'] =  vkm['MJ/vkm_fuel_PHEV_TNO']*(1-vkm['vkms_elec_TNO'])*gCO2_per_MJ_diesel #Fuel_per_non_electric km*Non_electric_kms*gCO2_per_MJ #diesel PHEV uses diesel #Note: CO2 is for comparison only, not used in article.

#Computation MJ/vkm from gCO2/vkm 
vkm.loc[(vkm['Fueltype_MPN'] == 1), 'SEC_TNO'] = vkm['CO2_TNO']/gCO2_per_MJ_petrol #Conversion factor that was used to convert original fuel use to CO2 emissions
vkm.loc[(vkm['Fueltype_MPN'] == 2), 'SEC_TNO'] = vkm['CO2_TNO']/gCO2_per_MJ_diesel #Conversion factor that was used to convert original fuel use to CO2 emissions
vkm.loc[(vkm['Fueltype_MPN'] == 4), 'SEC_TNO'] = vkm['CO2_TNO']/gCO2_per_MJ_petrol #Hybrids typically use gasoline

#MJ/vkm BEVs and PHEVs from Ruiter et al. 2021 data, as imputed using the car-models in the RDW-code
vkm.loc[vkm['Fueltype_MPN'] == 5, 'SEC_TNO'] = vkm['MJ/vkm_EVs_TNO']
vkm.loc[vkm['Fueltype_MPN'] == 15, 'SEC_TNO'] = vkm['MJ/vkm_EVs_TNO'] #This is already weighted by elec and petrol use #Does not affect CO2 calcs as those are based on original MJ/vkm_fuel_PHEV values only
vkm.loc[vkm['Fueltype_MPN'] == 25, 'SEC_TNO'] = vkm['MJ/vkm_EVs_TNO']

gasoline = vkm[vkm['Fueltype_MPN'] == 1]
vkm.loc[vkm['Fueltype_MPN'] == 7, 'SEC_TNO'] = gasoline['SEC_TNO'].mean()*(1.24/2.21) #Hydrogen cars use 1.24 MJ/vkm vs 2.21 MJ/vkm for gasoline (E10) cars in 2020 according to STREAM 2023 (webtool)

#For comparison purposes (Boxplot)
vkm.loc[(vkm['Fueltype_MPN'] == 1), 'SEC_RDW'] = vkm['Fuel_RDW_combined']*MJ_per_liter_petrol/100 #From L/100 vkm to MJ/vkm
vkm.loc[(vkm['Fueltype_MPN'] == 2), 'SEC_RDW'] = vkm['Fuel_RDW_combined']*MJ_per_liter_diesel/100 #EV consumption would be based on WLTP values: include for completeness or exclude cause unfair?
vkm.loc[(vkm['Fueltype_MPN'] == 3), 'SEC_RDW'] = vkm['Fuel_RDW_combined']*MJ_per_liter_LPG/100 #PHEVs are not included in Fuel_RDW_combined regardless
vkm.loc[(vkm['Fueltype_MPN'] == 4), 'SEC_RDW'] = vkm['Fuel_RDW_combined']*MJ_per_liter_petrol/100


'''XXXXX Dealing with the LPGs (fueltype 3): average weight and fuel use of LPG car models in Travelcard data XXXXX '''
# #Commented out to avoid very long loading times (the RDW datasets contain all cars in the Netherlands)

# import re 

# #Loading the huge RDW datasets
# RDW_Kentekens = pd.read_csv('Original_Data/Open_Data_RDW__Gekentekende_voertuigen.csv', dtype={'Massa rijklaar': 'float64'}, usecols = ['Kenteken','Merk','Handelsbenaming','Uitvoering','Massa rijklaar','Massa ledig voertuig'])
# RDW_Kentekens['Plate_RDWgeneral'] = RDW_Kentekens['Kenteken'].str.upper() #Just in case

# RDW_Brandstof = pd.read_csv('Original_Data/Open_Data_RDW__Gekentekende_voertuigen_brandstof.csv', usecols = ['Kenteken','Brandstof omschrijving'])
# RDW_Brandstof['Plate_RDWfuel'] = RDW_Brandstof['Kenteken'].str.upper() #Just in case
# RDW_Brandstof = RDW_Brandstof[RDW_Brandstof['Brandstof omschrijving'] == 'LPG']
# RDW_Brandstof.loc[RDW_Brandstof['Brandstof omschrijving'] == 'LPG', 'Brandstof omschrijving'] = 'Gas' 

# #Merging the RDW datasets
# RDW = pd.merge(RDW_Brandstof, RDW_Kentekens, left_on = 'Plate_RDWfuel', right_on = 'Plate_RDWgeneral', how = 'left') #
# RDW.loc[RDW['Massa rijklaar'].isna(), 'Massa rijklaar'] = RDW['Massa ledig voertuig'] + 100 
# RDW.loc[RDW['Massa rijklaar'] < 500, 'Massa rijklaar'] = RDW['Massa ledig voertuig'] + 100 
# RDW.loc[RDW['Massa rijklaar'] > 4000, 'Massa rijklaar'] = RDW['Massa ledig voertuig'] + 100 

# #Trying to synthesize the model-name
# RDW.loc[RDW['Handelsbenaming'].str.split(' ').str[0] == RDW['Merk'], 'Handelsbenaming'] = RDW['Handelsbenaming'].str.split(n=1).str[1] #Some model names start with a repetition of the general car brand #Code splits the handelsbenaming and then only saves the second part if the first part is equal to the merk
# RDW.loc[RDW['Handelsbenaming'].str.rstrip('0123456789') == RDW['Merk'], 'Handelsbenaming'] = RDW['Handelsbenaming'].str.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') #Some model names start with the repetition of the general car brand with a number (Mazda Mazda5) #Code looks if the handelsbenaming without numbers at the end is equal to the merk and in that case removes the letters (only saves the numbers at the end)
# RDW['Full_carmodel_name'] = RDW['Merk'] + ' ' + RDW['Handelsbenaming']  #The Full_carmodel_name is the brand plus the model name with white spaces
# RDW['Full_carmodel_name'] = RDW['Full_carmodel_name'] + ' ' + RDW['Brandstof omschrijving'] 
# RDW['Full_carmodel_name'] = RDW['Full_carmodel_name'].str.upper()

# RDW_kg_mean = RDW.groupby('Full_carmodel_name', as_index=False)['Massa rijklaar'].mean() 
# RDW_kg_mean.columns = ['Full_carmodel_name','kg_mean']
# RDW = pd.merge(RDW, RDW_kg_mean, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left')

# RDW = RDW[['Full_carmodel_name','kg_mean']]
# RDW = RDW.drop_duplicates()

# #Loading the praktijkverbruik data
# Praktijkverbruik = pd.read_csv('Original_Data/Praktijkverbruik/Praktijkverbruik_allyears.csv', sep = ',', index_col = 0) 
# Praktijk_LPGs = Praktijkverbruik[Praktijkverbruik['brandstof'] == 'Gas'] #2087 entries
# Praktijk_LPGs['Age_in_2020'] = 2020 - Praktijk_LPGs['Car_year']

# Praktijk_LPGs['Full_carmodel_name'] = Praktijk_LPGs['Merk'] + ' ' + Praktijk_LPGs['Model'] + ' ' + Praktijk_LPGs['brandstof'] 
# Praktijk_LPGs = Praktijk_LPGs[['Full_carmodel_name','Norm verbruik','Travelcard verbruik','CO2 fabriek','CO2 werkelijk','Age_in_2020']]
# Praktijk_LPGs.columns = ['Full_carmodel_name','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard', 'Age_in_2020']

# Praktijk_LPGs['Full_carmodel_name'] = [re.sub('[\(\[].*?[\)\]]', '', s) for s in Praktijk_LPGs['Full_carmodel_name'].tolist()] #Removing stuff between brackets: Audi A4 (4d) -> Audi A4
# Praktijk_LPGs = Praktijk_LPGs.replace(to_replace = '  ', value = ' ', regex = True) #Fix double white spaces (before computing means because for Dacia Sandero, some entries have a double white and some do not)
# Praktijk_LPGs['Full_carmodel_name'] = Praktijk_LPGs['Full_carmodel_name'].str.upper()

# Praktijk_LPGs_fuel_mean = Praktijk_LPGs.groupby('Full_carmodel_name', as_index=False)['Fuel_real_travelcard'].mean()
# Praktijk_LPGs_fuel_mean.columns = ['Full_carmodel_name','fuel_mean']
# Praktijk_LPGs = Praktijk_LPGs.merge(Praktijk_LPGs_fuel_mean, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left')

# Praktijk_age_mean = Praktijk_LPGs.groupby('Full_carmodel_name', as_index=False)['Age_in_2020'].mean() #There are many different car weights per model. Let's get the mean weight.
# Praktijk_age_mean.columns = ['Full_carmodel_name','Praktijk_age_mean']
# Praktijk_LPGs = pd.merge(Praktijk_LPGs, Praktijk_age_mean, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left')

# Praktijk_LPGs = Praktijk_LPGs[['Full_carmodel_name','fuel_mean','Praktijk_age_mean']]

# Praktijk_LPGs = Praktijk_LPGs.merge(RDW, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left') 
# Praktijk_LPGs = Praktijk_LPGs.drop_duplicates() #64 carmodels (different building years per model ignored)
# Praktijk_LPGs = Praktijk_LPGs[Praktijk_LPGs['kg_mean'] > 500] #44 of which could be coupled to RDW mass data
# Praktijk_LPGs.to_csv('Praktijkverbruik_LPGs_plus_RDW_data.csv')


'''XXXXX Computing LPG energy use (fueltype 3): simple linear model based on weight versus fuel use of carmodels XXXXX ''' #Note, there are only ca. fifty LPGs among the 3498 included vehicles. 

Praktijk_LPGs = pd.read_csv('Praktijkverbruik_LPGs_plus_RDW_data.csv', index_col = 0)
Praktijk_LPGs['SEC_mean'] = Praktijk_LPGs['fuel_mean']*MJ_per_liter_LPG/100

lm = LinearRegression() 
lm.fit(Praktijk_LPGs[['kg_mean']], Praktijk_LPGs['SEC_mean']) #Simple linear model of SEC vs weight in kg, average result is in line with STREAM (2023)
coefs = lm.coef_

vkm.loc[(vkm['Fueltype_MPN'] == 3), 'SEC_TNO'] = lm.intercept_ + lm.coef_[0]*vkm['Car_kg_RDW'] 


'''XXXXX Coupling, computing CO2 emissions, defining Car Types, Registering CarValid, and saving XXXXX'''

vkm = vkm[['CARID','SEC_TNO','CO2_TNO','SEC_RDW']] #Only keeping the relevant new columns
MPN3years = MPN3years.merge(vkm, left_on = 'CARID', right_on = 'CARID', how = 'left')

MPN3years.loc[(MPN3years['Fueltype_MPN'] == 1), 'SEC_travelcard_real'] = MPN3years['Fuel_real_travelcard']*MJ_per_liter_petrol/100 #From L/100 vkm to MJ/vkm
MPN3years.loc[(MPN3years['Fueltype_MPN'] == 2), 'SEC_travelcard_real'] = MPN3years['Fuel_real_travelcard']*MJ_per_liter_diesel/100 
MPN3years.loc[(MPN3years['Fueltype_MPN'] == 3), 'SEC_travelcard_real'] = MPN3years['Fuel_real_travelcard']*MJ_per_liter_LPG/100
MPN3years.loc[(MPN3years['Fueltype_MPN'] == 4), 'SEC_travelcard_real'] = MPN3years['Fuel_real_travelcard']*MJ_per_liter_petrol/100
MPN3years['SEC_Travelcard_TNO'] = MPN3years['SEC_travelcard_real']
MPN3years.loc[MPN3years['SEC_Travelcard_TNO'].isna(), 'SEC_Travelcard_TNO'] = MPN3years['SEC_TNO']

MPN3years = MPN3years[['HHID','CARID', 'HH_VALID', 'HH_weight', 'FracAdultHighedu', 'FracAdultWorkers', 'HH_under12', 'HH_12to17', 'HH_adults', 'HH_18to39', 'HH_40to59', 'HH_60plus', 'HH_Dutch', 'FracAdultMales', 'HH_inc', 'HH_inc_real', 'Density_PC5', 'km_super', 'km_highway', 'km_station', 'km_bigstation', 'km_center', 'km_mediumcenter', 'km_largercenter', 'km_hugecenter', 'NDVI', 'Landuse', 'f_4plus', 'km_bus', 'Tram_1000m', 'Parkingspot', 'HH_cars', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_design_MPN', 'Car_year_MPN', 'Fueltype_MPN', 'Car_kg_RDW', 'SEC_TNO', 'CO2_TNO', 'SEC_RDW', 'SEC_travelcard_real', 'SEC_Travelcard_TNO']]

#Computing CO2 emissions using the STREAM (2023) report and webtool: https://tools.ce.nl/stream/
MPN3years.loc[MPN3years['Fueltype_MPN'] == 1, 'TTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*69.5 
MPN3years.loc[MPN3years['Fueltype_MPN'] == 2, 'TTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*68.8 
MPN3years.loc[MPN3years['Fueltype_MPN'] == 3, 'TTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*67.5 
MPN3years.loc[MPN3years['Fueltype_MPN'] == 4, 'TTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*69.5 
MPN3years.loc[MPN3years['Fueltype_MPN'] == 5, 'TTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*0

MPN3years.loc[MPN3years['Fueltype_MPN'] == 1, 'WTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*91.9 
MPN3years.loc[MPN3years['Fueltype_MPN'] == 2, 'WTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*92.3 
MPN3years.loc[MPN3years['Fueltype_MPN'] == 3, 'WTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*76.0 
MPN3years.loc[MPN3years['Fueltype_MPN'] == 4, 'WTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*91.9
MPN3years.loc[MPN3years['Fueltype_MPN'] == 5, 'WTW_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*95.6

MPN3years.loc[MPN3years['Fueltype_MPN'] == 1, 'Total_CO2_Travelcard_TNO'] = MPN3years['WTW_CO2_Travelcard_TNO_EFfromSTREAM'] + 37
MPN3years.loc[MPN3years['Fueltype_MPN'] == 2, 'Total_CO2_Travelcard_TNO'] = MPN3years['WTW_CO2_Travelcard_TNO_EFfromSTREAM'] + 22
MPN3years.loc[MPN3years['Fueltype_MPN'] == 3, 'Total_CO2_Travelcard_TNO'] = np.nan
MPN3years.loc[MPN3years['Fueltype_MPN'] == 4, 'Total_CO2_Travelcard_TNO'] = np.nan
MPN3years.loc[MPN3years['Fueltype_MPN'] == 5, 'Total_CO2_Travelcard_TNO'] = MPN3years['WTW_CO2_Travelcard_TNO_EFfromSTREAM'] + 80

MPN3years.loc[MPN3years['Fueltype_MPN'] == 1, 'WTW2030_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*92.0
MPN3years.loc[MPN3years['Fueltype_MPN'] == 2, 'WTW2030_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*92.8
MPN3years.loc[MPN3years['Fueltype_MPN'] == 3, 'WTW2030_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*76.0
MPN3years.loc[MPN3years['Fueltype_MPN'] == 4, 'WTW2030_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*92.0
MPN3years.loc[MPN3years['Fueltype_MPN'] == 5, 'WTW2030_CO2_Travelcard_TNO_EFfromSTREAM'] = MPN3years['SEC_Travelcard_TNO']*28.8

#Creating Car Type and CarValid variables #Separating LPGs/unknown fuel types not worth it: few vehicles and removing them barely affects the standard vehicle categories
MPN3years['Type'] = 1 #Standard fuel type
MPN3years.loc[MPN3years['Fueltype_MPN'] == 2, 'Type'] = 2 #Diesel
MPN3years.loc[MPN3years['Fueltype_MPN'] == 4, 'Type'] = 5 #HEVs
MPN3years.loc[MPN3years['Fueltype_MPN'] == 5, 'Type'] = 5 #BEVs
MPN3years.loc[MPN3years['Fueltype_MPN'] == 7, 'Type'] = 5 #H2s
MPN3years.loc[MPN3years['Fueltype_MPN'] == 15, 'Type'] = 5 #Gasoline PHEVs
MPN3years.loc[MPN3years['Fueltype_MPN'] == 25, 'Type'] = 5 #Diesel PHEVs

MPN3years.loc[(MPN3years['Type'] == 1) & (MPN3years['Car_kg_RDW'] >= 1500), 'Type'] = 14 #Based on RDW massa rijklaar 
MPN3years.loc[(MPN3years['Type'] == 1) & (MPN3years['Car_kg_RDW'] >= 1250), 'Type'] = 13 
MPN3years.loc[(MPN3years['Type'] == 1) & (MPN3years['Car_kg_RDW'] >= 1000), 'Type'] = 12 
MPN3years.loc[(MPN3years['Type'] == 1) & (MPN3years['Car_kg_RDW'] < 1000), 'Type'] = 11

MPN3years.loc[(MPN3years['Type'] == 2) & (MPN3years['Car_kg_RDW'] >= 1500), 'Type'] = 24 #Based on RDW massa rijklaar
MPN3years.loc[(MPN3years['Type'] == 2) & (MPN3years['Car_kg_RDW'] >= 1250), 'Type'] = 23 
MPN3years.loc[(MPN3years['Type'] == 2) & (MPN3years['Car_kg_RDW'] < 1250), 'Type'] = 22  #Only one diesel weighs less than 1000kg

#Marking cars for which the fuel type, weight, or car year are unknown or unrealistic (except HEVs, for which weight and car year matter less)
#Note: you cannot remove these cars (that will crash the model), but they will be excluded from the cartype model component using the CarValid indicator
MPN3years['CarValid'] = 0
MPN3years.loc[(MPN3years['Fueltype_MPN'] > 0) & (MPN3years['Car_year_MPN'] >= 0) & (MPN3years['Car_kg_RDW'] >= 500), 'CarValid'] = 1 #Three cars are unrealistically light
MPN3years.loc[MPN3years['Type'] == 5, 'CarValid'] = 1 #Keep all HEVs due to their future relevance
MPN3years.loc[MPN3years['HH_cars'] == 0, 'CarValid'] = 0 #To be sure
MPN3years['Car_kg_RDW'].replace(to_replace = np.nan, value = 0, inplace = True) 

MPN3years.to_csv(data_directory + os.sep + 'MPN_processed_data/MPN3years_1entrypercar.csv')
