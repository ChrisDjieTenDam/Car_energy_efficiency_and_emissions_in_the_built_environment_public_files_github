# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:22:26 2021  by Chris ten Dam.
This code has been written for an academic study titled "Car energy efciency and emissions in the built environment".

It cleans the data from the Netherlands Vehicle Authority (RDW).
Next, it adds information on the energy use of PHEVs and BEVs from TNO publications.
Finally, it cleans data on real-world energy use from Travelcard and couples it based on the car-model, fuel type, and building year.
"""

import os

data_directory = r"Q:\research-driving-energy"

import pandas as pd
import numpy as np

'''XXXXX Importing and cleaning the RDW datasets XXXXX''' #Takes a lot of time due to huge RDW datasets, so added datasaving loop and cancelled out the code. 

# MPNKentekens2017 = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/Plate_private2017.csv', index_col = 0)
# MPNKentekens2018 = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/Plate_private2018.csv', index_col = 0)
# MPNKentekens2019 = pd.read_csv(data_directory + os.sep + 'MPN2019_originaldata/Plate_private2019.csv', index_col = 0)
# MPNKentekens = pd.concat([MPNKentekens2017, MPNKentekens2018, MPNKentekens2019]) #13047

# for Kenteken in ['KENTEKEN1', 'KENTEKEN2', 'KENTEKEN3', 'KENTEKEN4', 'KENTEKEN5', 'KENTEKEN6']: #KENTEKEN7 and KENTEKEN8 are all nan
#     MPNKentekens[Kenteken] = MPNKentekens[Kenteken].str.upper()

# #Importing and coupling RDW-data on the cars associated with those license plates #Nominaal continu maximumvermogen is only defined for EVs
# MPNKentekens1 = MPNKentekens[['KENTEKEN1','AUTO1_BOUWJAAR']] #Also gets rid of HHID and JAAR columns
# MPNKentekens2 = MPNKentekens[['KENTEKEN2','AUTO2_BOUWJAAR']]
# MPNKentekens3 = MPNKentekens[['KENTEKEN3','AUTO3_BOUWJAAR']]
# MPNKentekens4 = MPNKentekens[['KENTEKEN4','AUTO4_BOUWJAAR']]
# MPNKentekens5 = MPNKentekens[['KENTEKEN5','AUTO5_BOUWJAAR']]
# MPNKentekens6 = MPNKentekens[['KENTEKEN6','AUTO6_BOUWJAAR']]

# RDWforMPN = pd.concat([MPNKentekens1,MPNKentekens2,MPNKentekens3,MPNKentekens4,MPNKentekens5,MPNKentekens6]) 

# RDWforMPN = RDWforMPN.reset_index(drop = True)
# RDWforMPN['Plate_CarData'] = RDWforMPN['KENTEKEN1'] #Without fuel data (Brandstof[Kenteken] does not match), you can still analyze vehicle type
# RDWforMPN.loc[RDWforMPN['Plate_CarData'].isna(), 'Plate_CarData'] = RDWforMPN['KENTEKEN2']
# RDWforMPN.loc[RDWforMPN['Plate_CarData'].isna(), 'Plate_CarData'] = RDWforMPN['KENTEKEN3']
# RDWforMPN.loc[RDWforMPN['Plate_CarData'].isna(), 'Plate_CarData'] = RDWforMPN['KENTEKEN4']
# RDWforMPN.loc[RDWforMPN['Plate_CarData'].isna(), 'Plate_CarData'] = RDWforMPN['KENTEKEN5']
# RDWforMPN.loc[RDWforMPN['Plate_CarData'].isna(), 'Plate_CarData'] = RDWforMPN['KENTEKEN6']
# RDWforMPN = RDWforMPN[RDWforMPN['Plate_CarData'].notna()] #If a household did not give the plate for any car
# RDWforMPN = RDWforMPN.drop_duplicates()

# RDW_Brandstof = pd.read_csv('Original_Data/Open_Data_RDW__Gekentekende_voertuigen_brandstof.csv')
# RDW_Brandstof['Plate_RDWfuel'] = RDW_Brandstof['Kenteken'].str.upper() #Just in case
# RDWforMPN = pd.merge(RDWforMPN, RDW_Brandstof, left_on = 'Plate_CarData', right_on = 'Plate_RDWfuel', how = 'left') 

# RDW_Kentekens = pd.read_csv('Original_Data/Open_Data_RDW__Gekentekende_voertuigen.csv', dtype={'Massa rijklaar': 'float64'}, usecols = ['Kenteken','Merk','Handelsbenaming','Uitvoering','Massa rijklaar','Massa ledig voertuig'])
# RDW_Kentekens['Plate_RDWgeneral'] = RDW_Kentekens['Kenteken'].str.upper() #Just in case
# RDWforMPN = pd.merge(RDWforMPN, RDW_Kentekens, left_on = 'Plate_CarData', right_on = 'Plate_RDWgeneral', how = 'left')

# RDWforMPN.loc[RDWforMPN['Massa rijklaar'].isna(), 'Massa rijklaar'] = RDWforMPN['Massa ledig voertuig'] + 100 
# RDWforMPN.loc[RDWforMPN['Massa rijklaar'] < 500, 'Massa rijklaar'] = RDWforMPN['Massa ledig voertuig'] + 100 
# RDWforMPN.loc[RDWforMPN['Massa rijklaar'] > 4000, 'Massa rijklaar'] = RDWforMPN['Massa ledig voertuig'] + 100 
# RDWforMPN.loc[RDWforMPN['CO2 uitstoot gecombineerd'].isna(), 'CO2 uitstoot gecombineerd'] = RDWforMPN['CO2 uitstoot gewogen'] #CO2 uitstoot gewogen = CO2 emissions of PHEVs (variable only defined for PHEVs)

# RDWforMPN['Car_year'] = RDWforMPN['AUTO1_BOUWJAAR'] #For coupling. The car-year for computations is added from MPN based on which car is used most. 
# RDWforMPN.loc[RDWforMPN['Car_year'].isna(), 'Car_year'] = RDWforMPN['AUTO2_BOUWJAAR']
# RDWforMPN.loc[RDWforMPN['Car_year'].isna(), 'Car_year'] = RDWforMPN['AUTO3_BOUWJAAR']
# RDWforMPN.loc[RDWforMPN['Car_year'].isna(), 'Car_year'] = RDWforMPN['AUTO4_BOUWJAAR']
# RDWforMPN.loc[RDWforMPN['Car_year'].isna(), 'Car_year'] = RDWforMPN['AUTO5_BOUWJAAR']
# RDWforMPN.loc[RDWforMPN['Car_year'].isna(), 'Car_year'] = RDWforMPN['AUTO6_BOUWJAAR']

# RDWforMPN = RDWforMPN[['Plate_CarData', 'Brandstof volgnummer', 'Brandstof omschrijving', 'Brandstofverbruik gecombineerd', 'CO2 uitstoot gecombineerd', 'CO2 uitstoot gewogen', 'Nettomaximumvermogen', 'Emissie co2 gecombineerd wltp', 'Brandstof verbruik gecombineerd wltp', 'Klasse hybride elektrisch voertuig', 'Massa rijklaar', 'Handelsbenaming', 'Uitvoering', 'Car_year', 'Merk']]
# RDWforMPN = RDWforMPN.sort_values('Plate_CarData')

# RDWforMPN.to_csv(data_directory + os.sep + 'MPN_processed_data/RDWforMPN3years.csv')


'''XXXXX Specifying model names, dropping double entries, and adding BEV and PHEV data from TNO XXXXX'''

CarData = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/RDWforMPN3years.csv', index_col = 0) #The license plates of the sample households, in alphabetical order without any household IDs or other identifyers. 

#Dealing with double entries due to bifueled vehicles #You could also use the brandstofvolgnummer provided for bi-fueled vehicles. This would be less consistent though and would lead to loss of information for NOVCs registered as electricity-first
CarData['Brandstof omschrijving'].replace(to_replace = 'CNG', value = 'A_CNG', inplace = True)
CarData['Brandstof omschrijving'].replace(to_replace = 'LPG', value = 'B_LPG', inplace = True) #For 15 CNG or LPG/Gasoline cars the LPG-values are not equivalent to the gasoline ones. LPG is cheaper than gasoline though and is also the unconventional fuel type. It is thus reasonable to assume that people will mostly use LPG in LPG-gasoline vehicles. 
CarData['Brandstof omschrijving'].replace(to_replace = 'Elektriciteit', value = 'Z_Elektriciteit', inplace = True) #Nonchargeable gasoline vehicles only use gasoline and I couple the electricity use of PHEVs and BEVs manually. Besides, the data on electricity for hybrid vehicles only displays the nominal max power
CarData = CarData.sort_values('Brandstof omschrijving') #Keep the LPG entries for bifueled LPG-gasoline vehicles and the gasoline entries for gasoline-electric vehicles
CarData = CarData.drop_duplicates(subset = 'Plate_CarData', keep = 'first') #Also drops 3 double license plate entries in the MPN-data

#To allow coupling Praktijkverbruik data per fueltype
CarData.loc[CarData['Brandstof omschrijving'] == 'Benzine', 'Fueltype_derivation'] = 'Benzine'
CarData.loc[CarData['Brandstof omschrijving'] == 'Diesel', 'Fueltype_derivation'] = 'Diesel'
CarData.loc[CarData['Brandstof omschrijving'] == 'B_LPG', 'Fueltype_derivation'] = 'Gas' 
CarData.loc[CarData['Brandstof omschrijving'] == 'A_CNG', 'Fueltype_derivation'] = 'Aardgas' 
CarData.loc[CarData['Brandstof omschrijving'] == 'Z_Elektriciteit', 'Fueltype_derivation'] = 'Elektrisch'
CarData.loc[CarData['Klasse hybride elektrisch voertuig'].notna(), 'Fueltype_derivation'] = 'Hybride'

#Trying to synthesize the model-name
CarData.loc[CarData['Handelsbenaming'].str.split(' ').str[0] == CarData['Merk'], 'Handelsbenaming'] = CarData['Handelsbenaming'].str.split(n=1).str[1] #Some model names start with a repetition of the general car brand #Code splits the handelsbenaming and then only saves the second part if the first part is equal to the merk
CarData.loc[CarData['Handelsbenaming'].str.rstrip('0123456789') == CarData['Merk'], 'Handelsbenaming'] = CarData['Handelsbenaming'].str.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') #Some model names start with the repetition of the general car brand with a number (Mazda Mazda5) #Code looks if the handelsbenaming without numbers at the end is equal to the merk and in that case removes the letters (only saves the numbers at the end)
CarData['Couple_year'] = CarData['Car_year']
CarData['Full_carmodel_name'] = CarData['Merk'] + ' ' + CarData['Handelsbenaming']  #The Full_carmodel_name is the brand plus the model name with white spaces

#Conversion factors for adding TNO data
MJperL_gasoline = 31.350 #0.75 kg/L E10 gasoline *41.8 MJ/kg. Source: STREAM 2023 (page 94)
MJperL_diesel = 35.952 #0.84 kg/L B7 diesel *42.8 MJ/kg. Source: STREAM 2023 (page 94)
MJperkWh_over100 = 3.6/100

#Manually determining energy use of BEVs in MPN except one FIAT 500E
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'MERCEDES-BENZ B 250 E'), 'MJ/vkm_EVs_TNO'] = 17.5*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'RENAULT ZOE'), 'MJ/vkm_EVs_TNO'] = 22.4*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'BMW I I3'), 'MJ/vkm_EVs_TNO'] = 19.7*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'TESLA MODEL 3'), 'MJ/vkm_EVs_TNO'] = 19.5*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'HYUNDAI IONIQ'), 'MJ/vkm_EVs_TNO'] = 20.0*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'TESLA MOTORS MODEL X'), 'MJ/vkm_EVs_TNO'] = 22.2*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'TESLA MODEL X'), 'MJ/vkm_EVs_TNO'] = 22.2*MJperkWh_over100 #The same vehicle
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'JAGUAR I-PACE'), 'MJ/vkm_EVs_TNO'] = 26.4*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'VOLKSWAGEN GOLF'), 'MJ/vkm_EVs_TNO'] = 17.8*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'KIA NIRO'), 'MJ/vkm_EVs_TNO'] = 16.2*MJperkWh_over100
CarData.loc[(CarData['Brandstof omschrijving'] == 'Z_Elektriciteit') & (CarData['Full_carmodel_name' ] == 'NISSAN LEAF 40KWH'), 'MJ/vkm_EVs_TNO'] = 18.6*MJperkWh_over100

#Manually determining energy use of PHEVs in MPN #Using real-data of share of electric kms per model (Gijlswijk et al. (2020), with #2022 number for reference) #Units of computation checked 20220831
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'FORD-CNG-TECHNIK C-MAX ENERGI'), 'vkms_elec_TNO'] =  0.222 #The car is registered as a gasoline PHEV  
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'MITSUBISHI OUTLANDER'), 'vkms_elec_TNO'] =  0.185 #Gijlswijk 2022: 0.223  
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'PORSCHE PANAMERA S E-HYBRID'), 'vkms_elec_TNO'] = 0.047# (0.223+0.249+0.204+0.179+0.336+0.119+0.214+0.214)/8
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'PORSCHE CAYENNE S E-HYBRID'), 'vkms_elec_TNO'] = 0.262
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLKSWAGEN GOLF'), 'vkms_elec_TNO'] = 0.225 #0.249
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLKSWAGEN PASSAT'), 'vkms_elec_TNO'] = 0.181 #0.204
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'AUDI A3 SPORTBACK E-TRON'), 'vkms_elec_TNO'] = 0.156 #0.179
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'OPEL AMPERA'), 'vkms_elec_TNO'] = 0.392 #0.336
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'TOYOTA PRIUS PLUG-IN HYBRID'), 'vkms_elec_TNO'] = 0.161 #0.119
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLVO V60 PLUG IN HYBRID'), 'vkms_elec_TNO'] = 0.208 #0.214 
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLVO V60 TWIN ENGINE'), 'vkms_elec_TNO'] = 0.138 #0.214

CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'FORD-CNG-TECHNIK C-MAX ENERGI'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/14.6 #MJ/L over km/L 
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'MITSUBISHI OUTLANDER'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/12.5
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'PORSCHE PANAMERA S E-HYBRID'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/11.5
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'PORSCHE CAYENNE S E-HYBRID'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/7.8
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLKSWAGEN GOLF'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/13.7
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLKSWAGEN PASSAT'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/14.1
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'AUDI A3 SPORTBACK E-TRON'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/15.2
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'OPEL AMPERA'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/13.6
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'TOYOTA PRIUS PLUG-IN HYBRID'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_gasoline/18.5
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLVO V60 PLUG IN HYBRID'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_diesel/14.7
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLVO V60 TWIN ENGINE'), 'MJ/vkm_fuel_PHEV_TNO'] = MJperL_diesel/15.2

CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'FORD-CNG-TECHNIK C-MAX ENERGI'), 'MJ/vkm_EVs_TNO'] =  CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 19.7*MJperkWh_over100*CarData['vkms_elec_TNO'] #The car is registered as a gasoline PHEV
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'MITSUBISHI OUTLANDER'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 23.1*MJperkWh_over100*CarData['vkms_elec_TNO'] #MJ/liter gedeeld door 12.5 vkm/liter voor 62% en 23.1 kWh/100km keer 3.6 MJ/kWh gedeeld door 100
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'PORSCHE PANAMERA S E-HYBRID'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 25.1*MJperkWh_over100*CarData['vkms_elec_TNO']
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'PORSCHE CAYENNE S E-HYBRID'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 36.9*MJperkWh_over100*CarData['vkms_elec_TNO']
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLKSWAGEN GOLF'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 21.1*MJperkWh_over100*CarData['vkms_elec_TNO']
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLKSWAGEN PASSAT'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 20.4*MJperkWh_over100*CarData['vkms_elec_TNO']
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'AUDI A3 SPORTBACK E-TRON'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 19.0*MJperkWh_over100*CarData['vkms_elec_TNO']
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'OPEL AMPERA'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 21.2*MJperkWh_over100*CarData['vkms_elec_TNO']
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'TOYOTA PRIUS PLUG-IN HYBRID'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 15.5*MJperkWh_over100*CarData['vkms_elec_TNO']
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLVO V60 PLUG IN HYBRID'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 27.4*MJperkWh_over100*CarData['vkms_elec_TNO'] #the vehicles are diesel according to RDW data
CarData.loc[(CarData['Klasse hybride elektrisch voertuig'] == 'OVC-HEV') & (CarData['Full_carmodel_name' ] == 'VOLVO V60 TWIN ENGINE'), 'MJ/vkm_EVs_TNO'] = CarData['MJ/vkm_fuel_PHEV_TNO'] * (1-CarData['vkms_elec_TNO']) + 26.5*MJperkWh_over100*CarData['vkms_elec_TNO'] #the vehicles are diesel according to RDW data

CarData['Full_carmodel_name'] = CarData['Full_carmodel_name'] + ' ' + CarData['Couple_year'].astype(str) + ' ' + CarData['Fueltype_derivation']
CarData['Full_carmodel_name'] = CarData['Full_carmodel_name'].str.upper()


'''XXXXX Uploading Travelcard data [from website] and cleaning car model names XXXXX'''
#"Uitvoering" seems to be a different column (to contain different data) in RDW and Praktijkverbruik. A number of couplings seem to fail because names in RDW either contain too much or too little information (data on the precise make is added or ommitted)

import re 

Praktijkverbruik = pd.read_csv('Original_Data/Praktijkverbruik/Praktijkverbruik_allyears.csv', sep = ',', index_col = 0) #65550 model-building year-subbrand combinations

Praktijkverbruik['Car_year'] = Praktijkverbruik['Car_year'].astype(float)
Praktijkverbruik['Full_carmodel_name'] = Praktijkverbruik['Merk'] + ' ' + Praktijkverbruik['Model'] + ' ' + Praktijkverbruik['Car_year'].astype(str) + ' ' + Praktijkverbruik['brandstof']
Praktijkverbruik = Praktijkverbruik[['Full_carmodel_name','Norm verbruik','Travelcard verbruik','CO2 fabriek','CO2 werkelijk']]
Praktijkverbruik.columns = ['Full_carmodel_name','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']

Praktijkverbruik['Full_carmodel_name'] = [re.sub('[\(\[].*?[\)\]]', '', s) for s in Praktijkverbruik['Full_carmodel_name'].tolist()] #Removing stuff between brackets: Audi A4 (4d) -> Audi A4
Praktijkverbruik = Praktijkverbruik.replace(to_replace = '  ', value = ' ', regex = True) #Fix double white spaces (before computing means because for Dacia Sandero, some entries have a double white and some do not)
Praktijkverbruik['Full_carmodel_name'] = Praktijkverbruik['Full_carmodel_name'].str.upper()


'''XXXXX Coupling the data XXXXX'''

Fuel_norm_accordingtotravelcard_median = Praktijkverbruik.groupby('Full_carmodel_name', as_index=False)['Fuel_norm_accordingtotravelcard'].median() #Checked for Alfa Romeo Giulietta 
Fuel_real_travelcard_median = Praktijkverbruik.groupby('Full_carmodel_name', as_index=False)['Fuel_real_travelcard'].median() #Use Median to reduce influence of extra light of heavy variants of car-models, which may otherwise cause systematic bias. 
CO2_norm_accordingtotravelcard_median = Praktijkverbruik.groupby('Full_carmodel_name', as_index=False)['CO2_norm_accordingtotravelcard'].median() 
CO2_real_travelcard_median = Praktijkverbruik.groupby('Full_carmodel_name', as_index=False)['CO2_real_travelcard'].median() 

CarData = pd.merge(CarData, Fuel_norm_accordingtotravelcard_median, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left')
CarData = pd.merge(CarData, Fuel_real_travelcard_median, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left')
CarData = pd.merge(CarData, CO2_norm_accordingtotravelcard_median, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left')
CarData = pd.merge(CarData, CO2_real_travelcard_median, left_on = 'Full_carmodel_name', right_on = 'Full_carmodel_name', how = 'left')

for Travelcard_var in ['Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']:
    CarData.loc[CarData['MJ/vkm_EVs_TNO'] > 0, Travelcard_var] = np.nan #Electric cars are given in kWh/100 km and PHEV in what? Better to use TNO data directly
    CarData.loc[CarData['Fuel_real_travelcard'] > 50, Travelcard_var] = np.nan #Dozens of vehicles have a huge fuel use and exactly 68 g/vkm CO2 emissions according to Travelcard: highly unlikely
    CarData.loc[CarData['CO2_real_travelcard'] == 68, Travelcard_var] = np.nan #Dozens of vehicles have a huge fuel use and exactly 68 g/vkm CO2 emissions according to Travelcard: highly unlikely

#Do not save things like the full-model name to protect privacy #Note: weighed emissions or fuel use in RDW data is only defined for PHEVs
CarData = CarData[['Plate_CarData','Massa rijklaar','Klasse hybride elektrisch voertuig','Brandstof omschrijving','Brandstofverbruik gecombineerd','CO2 uitstoot gecombineerd','Nettomaximumvermogen','Emissie co2 gecombineerd wltp','Brandstof verbruik gecombineerd wltp','vkms_elec_TNO','MJ/vkm_EVs_TNO','MJ/vkm_fuel_PHEV_TNO','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']]#, 'Brandstof volgnummer', 'Brandstofverbruik stad','Brandstofverbruik buiten de stad', 'Netto max vermogen elektrisch', 'Klasse hybride elektrisch voertuig'
CarData.columns = ['Plate_CarData','Car_kg_RDW','Hybrid_type_RDW','Fueltype_str_RDW','Fuel_RDW_combined','CO2_RDW_combined','Nettmaxpower_RDW','CO2_RDW_WLTP_combined','Fuel_RDW_WLTP_combined','vkms_elec_TNO','MJ/vkm_EVs_TNO','MJ/vkm_fuel_PHEV_TNO','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']
CarData.to_csv(data_directory + os.sep + 'MPN_processed_data/CarData3years.csv')
