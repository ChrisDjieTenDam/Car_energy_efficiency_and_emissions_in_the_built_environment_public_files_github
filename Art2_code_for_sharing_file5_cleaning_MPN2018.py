# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:06:12 2021 by Chris ten Dam.
This code has been written for an academic study titled "Driving energy and emissions: the built environment influence on car efficiency".

It cleans the data from the 2018 Netherlands Mobility Panel.
There are multiple datasets: one for each survey.

After cleaning and combining the relevant data, the privacy-sensitive data on household incomes and residential environment is added.
Finally, the identifyers (license plates and PC6-addresses) are dropped, along with all other columns that will not be used immediately. 
"""

import os

data_directory = r"Q:\research-driving-energy"

import numpy as np
import pandas as pd


'''XXXXX Preprocessing the Personal data questionnaires XXXXX '''

P = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/P2018.csv', usecols = ['HHID','HH_VALID','PERSID','GESLACHT','HERKOMST_w5','WERKSITUATIE_MEEST_w5','WERKURENRECENT_CONTINU','OPLEIDING','KLEEFT2'])

P['OPLEIDING'].replace(to_replace = 8, value = 7, inplace = True) #Higher education (HBO & WO) according to CBS classification
P.loc[P['OPLEIDING'] == 7, 'Highereducated'] = 1 
HH_highereducated = P.groupby('HHID', as_index=False)['Highereducated'].sum()
HH_highereducated.columns = ['HHID','HH_highereducated']
P = pd.merge(P, HH_highereducated, left_on='HHID', right_on='HHID', how = 'left')

###Societal position categories
P['SP'] = P['WERKSITUATIE_MEEST_w5'] 
for other in [4,5,8,9,10,99]: #Keep to prevent part time workers to be misclassified as fulltime workers
    P.loc[P['SP'] == other, 'SP'] = 11 
P.loc[(P['WERKURENRECENT_CONTINU'] < 24) & (P['SP'] != 6) & (P['SP'] != 7), 'SP'] = 11 #Working less than three days a week to other

P.loc[(P['SP'] == 1)|(P['SP'] == 2)|(P['SP'] == 3), 'SP'] = 1 #3days/week self-employed, employed by private organization, or employed by government
P.loc[(P['KLEEFT2'] >= 3) & (P['SP'] == 1), 'AdultWorker'] = 1
HH_AdultWorkers = P.groupby('HHID', as_index=False)['AdultWorker'].sum()
HH_AdultWorkers.columns = ['HHID','HH_AdultWorkers']
P = pd.merge(P, HH_AdultWorkers, left_on='HHID', right_on='HHID', how = 'left')

###Migrant backgrounds
P.loc[P['HERKOMST_w5'] == 1, 'Dutch'] = 1
HH_Dutch = P.groupby('HHID', as_index = False)['Dutch'].sum()
HH_Dutch.columns = ['HHID','HH_Dutch']
P = pd.merge(P, HH_Dutch, left_on = 'HHID', right_on = 'HHID', how = 'left')

###Age categories
P.loc[P['KLEEFT2'] == 2, '12to17'] = 1 
HH_12to17 = P.groupby('HHID', as_index=False)['12to17'].sum()
HH_12to17.columns = ['HHID','HH_12to17']
P = pd.merge(P, HH_12to17, left_on='HHID', right_on='HHID', how = 'left')

P.loc[(P['KLEEFT2'] >= 3) & (P['KLEEFT2'] <= 5), '18to39'] = 1 
HH_18to39 = P.groupby('HHID', as_index=False)['18to39'].sum()
HH_18to39.columns = ['HHID','HH_18to39']
P = pd.merge(P, HH_18to39, left_on='HHID', right_on='HHID', how = 'left')

P.loc[(P['KLEEFT2'] >= 6) & (P['KLEEFT2'] <= 7), '40to59'] = 1 
HH_40to59 = P.groupby('HHID', as_index=False)['40to59'].sum()
HH_40to59.columns = ['HHID','HH_40to59']
P = pd.merge(P, HH_40to59, left_on='HHID', right_on='HHID', how = 'left')

P.loc[P['KLEEFT2'] >= 8, '60plus'] = 1 
HH_60plus = P.groupby('HHID', as_index=False)['60plus'].sum()
HH_60plus.columns = ['HHID','HH_60plus']
P = pd.merge(P, HH_60plus, left_on='HHID', right_on='HHID', how = 'left')

P.loc[(P['KLEEFT2'] >= 3) & (P['GESLACHT'] == 1), 'AdultMale'] = 1
HH_AdultMales = P.groupby('HHID', as_index = False)['AdultMale'].sum()
HH_AdultMales.columns = ['HHID','HH_AdultMales']
P = pd.merge(P, HH_AdultMales, left_on = 'HHID', right_on = 'HHID', how = 'left')

P = P[['HHID','HH_VALID','PERSID','HH_highereducated','HH_AdultWorkers','HH_Dutch','HH_12to17','HH_18to39','HH_40to59','HH_60plus','HH_AdultMales']]


'''XXXXX Preprocessing the Household data questionnaires XXXXX '''

HH = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/HH2018.csv', usecols = ['HHID','JAAR','HHPERS','N_KIND','HHAUTO_N','HHPARK1','bushalte1xpu','tramhalte']) 

HH['bushalte1xpu'] = HH['bushalte1xpu']/1000 #meters to kilometers
HH['Tram_1000m'] = 0
HH.loc[(HH['tramhalte'] <= 1000), 'Tram_1000m'] = 1 #Because just distance to tram becomes a proxy for distance to Randstad

HH.loc[HH['HHPARK1'] == 99, 'HHPARK1'] = 0 #"no household questionnaire, no imputation from 2017 possible"

HH = HH[['HHID','JAAR','HHAUTO_N','HHPERS','N_KIND','HHPARK1','bushalte1xpu','Tram_1000m']]


'''XXXXX Preprocessing the Car data questionnaires  and adding (privacy_sesitive) plates XXXXX ''' 

Car = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/Car2018.csv')

#Only keep cars which are known to exist #There some cars with "gasoline" or "other" fuel type with AUTO1 = 99 or 0, this code assumes that these are data errors
for auto in ['AUTO1','AUTO2','AUTO3','AUTO4','AUTO5']:
    Car[auto].replace(to_replace = 99, value = np.nan, inplace = True)
    Car[auto].replace(to_replace = 0, value = np.nan, inplace = True) 

#Find the highest distance category (vkm/year) for each household #AUTO_1KM 2017 matches categories in 2018 and 2019 (same limits)
for km in ['AUTO1_KM','AUTO2_KM','AUTO3_KM','AUTO4_KM','AUTO5_KM']: #Replace with ones instead of nans cause otherwise, people with only 1 car with a 97 or 6 value get ignored (inter alia reducing the number of EVs)
    Car.loc[Car[km] >= 6, km] = np.nan #"don't know/won't say", though it is arguable that these cars are actually used a lot
    Car.loc[Car[km].isna(), km] = Car[km].mean()

#Adding the car license plates and building years #PRIVACY-SENSITIVE MERGER
Plate = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/Plate_private2018.csv', index_col = 0)
Car = pd.merge(Car, Plate, left_on = 'HHID', right_on = 'HHID', how = 'left')
for plate in ['KENTEKEN1','KENTEKEN2','KENTEKEN3','KENTEKEN4','KENTEKEN5']:
    Car[plate] = Car[plate].str.upper() #Cause some plates are in small letters

Vars = pd.read_csv('Mostusedcar_model_variables.csv')

for jaar in ['KAUTO1_BOUWJAAR','KAUTO2_BOUWJAAR','KAUTO3_BOUWJAAR','KAUTO4_BOUWJAAR','KAUTO5_BOUWJAAR']: #because cars built before 2000 are categorized in 2020 (>20 years old)
    Car.loc[Car[jaar] == 1, jaar] = 1995 #1990-2000: 25 years old
    Car.loc[Car[jaar] == 2, jaar] = 1985 
    
Car['age1'] = 2018 - Car['KAUTO1_BOUWJAAR']
Car['age2'] = 2018 - Car['KAUTO2_BOUWJAAR']
Car['age3'] = 2018 - Car['KAUTO3_BOUWJAAR']
Car['age4'] = 2018 - Car['KAUTO4_BOUWJAAR']
Car['age5'] = 2018 - Car['KAUTO5_BOUWJAAR']

for leeftijd in ['age1','age2','age3','age4','age5']:
     Car.loc[Car[leeftijd].isna(), leeftijd] = Car[leeftijd].mean() #So that nonvalid cars do not get assigned a zero age (and thus a very high probability)
     
Car = Car.fillna(0)

Car['logit_1'] = Vars['KM_category'].item()*Car['AUTO1_KM'] + Vars['Age'].item()*Car['age1'] 
Car['logit_2'] = Vars['KM_category'].item()*Car['AUTO2_KM'] + Vars['Age'].item()*Car['age2'] 
Car['logit_3'] = Vars['KM_category'].item()*Car['AUTO3_KM'] + Vars['Age'].item()*Car['age3'] 
Car['logit_4'] = Vars['KM_category'].item()*Car['AUTO4_KM'] + Vars['Age'].item()*Car['age4'] 
Car['logit_5'] = Vars['KM_category'].item()*Car['AUTO5_KM'] + Vars['Age'].item()*Car['age5'] 

#Model applied for all cars in multicar households. Coefficients based on all cars in multicar households in 2020.
Car['Prob_car1isusedmost'] = 1 #For all single-car households

Sum_exp_2cars = np.e**(Car['logit_1']) + np.e**(Car['logit_2'])
Car.loc[Car['HHAUTO_N'] == 2, 'Prob_car1isusedmost'] = np.e**(Car['logit_1'])/Sum_exp_2cars
Car.loc[Car['HHAUTO_N'] == 2, 'Prob_car2isusedmost'] = np.e**(Car['logit_2'])/Sum_exp_2cars

Sum_exp_3cars = np.e**(Car['logit_1']) + np.e**(Car['logit_2']) + np.e**(Car['logit_3'])
Car.loc[Car['HHAUTO_N'] == 3, 'Prob_car1isusedmost'] = np.e**(Car['logit_1'])/Sum_exp_3cars
Car.loc[Car['HHAUTO_N'] == 3, 'Prob_car2isusedmost'] = np.e**(Car['logit_2'])/Sum_exp_3cars
Car.loc[Car['HHAUTO_N'] == 3, 'Prob_car3isusedmost'] = np.e**(Car['logit_3'])/Sum_exp_3cars

Sum_exp_4cars = np.e**(Car['logit_1']) + np.e**(Car['logit_2']) + np.e**(Car['logit_3']) + np.e**(Car['logit_4'])
Car.loc[Car['HHAUTO_N'] == 4, 'Prob_car1isusedmost'] = np.e**(Car['logit_1'])/Sum_exp_4cars
Car.loc[Car['HHAUTO_N'] == 4, 'Prob_car2isusedmost'] = np.e**(Car['logit_2'])/Sum_exp_4cars
Car.loc[Car['HHAUTO_N'] == 4, 'Prob_car3isusedmost'] = np.e**(Car['logit_3'])/Sum_exp_4cars
Car.loc[Car['HHAUTO_N'] == 4, 'Prob_car4isusedmost'] = np.e**(Car['logit_4'])/Sum_exp_4cars

Sum_exp_5cars = np.e**(Car['logit_1']) + np.e**(Car['logit_2']) + np.e**(Car['logit_3']) + np.e**(Car['logit_4']) + np.e**(Car['logit_5'])
Car.loc[Car['HHAUTO_N'] >= 5, 'Prob_car1isusedmost'] = np.e**(Car['logit_1'])/Sum_exp_5cars #If a household has 6 cars, we simply ignore car number 6
Car.loc[Car['HHAUTO_N'] >= 5, 'Prob_car2isusedmost'] = np.e**(Car['logit_2'])/Sum_exp_5cars
Car.loc[Car['HHAUTO_N'] >= 5, 'Prob_car3isusedmost'] = np.e**(Car['logit_3'])/Sum_exp_5cars
Car.loc[Car['HHAUTO_N'] >= 5, 'Prob_car4isusedmost'] = np.e**(Car['logit_4'])/Sum_exp_5cars
Car.loc[Car['HHAUTO_N'] >= 5, 'Prob_car5isusedmost'] = np.e**(Car['logit_5'])/Sum_exp_5cars

Car1 = Car[Car['AUTO1'] == 1]
Car1 = Car1[['HHID','HHAUTO_N', 'AUTO1', 'AUTO1_BRANDSTOF_A_w6', 'AUTO1_BRANDSTOF_B_w6', 'Prob_car1isusedmost', 'AUTO1_AANSCHAF', 'AUTO1_GEWLEEG','AUTO1_INRICHT','KENTEKEN1','AUTO1_BOUWJAAR']]
Car1.columns = ['HHID','HH_cars', 'AUTO1', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car2 = Car[Car['AUTO2'] == 1]
Car2 = Car2[['HHID','HHAUTO_N', 'AUTO2', 'AUTO2_BRANDSTOF_A_w6', 'AUTO2_BRANDSTOF_B_w6', 'Prob_car2isusedmost', 'AUTO2_AANSCHAF', 'AUTO2_GEWLEEG','AUTO2_INRICHT','KENTEKEN2','AUTO2_BOUWJAAR']]
Car2.columns = ['HHID','HH_cars', 'AUTO2', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car3 = Car[Car['AUTO3'] == 1]
Car3 = Car3[['HHID','HHAUTO_N', 'AUTO3', 'AUTO3_BRANDSTOF_A_w6', 'AUTO3_BRANDSTOF_B_w6', 'Prob_car3isusedmost', 'AUTO3_AANSCHAF', 'AUTO3_GEWLEEG','AUTO3_INRICHT','KENTEKEN3','AUTO3_BOUWJAAR']]
Car3.columns = ['HHID','HH_cars', 'AUTO3', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car4 = Car[Car['AUTO4'] == 1] 
Car4 = Car4[['HHID','HHAUTO_N', 'AUTO4', 'AUTO4_BRANDSTOF_A_w6', 'AUTO4_BRANDSTOF_B_w6', 'Prob_car4isusedmost', 'AUTO4_AANSCHAF', 'AUTO4_GEWLEEG','AUTO4_INRICHT','KENTEKEN4','AUTO4_BOUWJAAR']]
Car4.columns = ['HHID','HH_cars', 'AUTO4', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car5 = Car[Car['AUTO5'] == 1] 
Car5 = Car5[['HHID','HHAUTO_N', 'AUTO5', 'AUTO5_BRANDSTOF_A_w6', 'AUTO5_BRANDSTOF_B_w6', 'Prob_car5isusedmost', 'AUTO5_AANSCHAF', 'AUTO5_GEWLEEG','AUTO5_INRICHT','KENTEKEN5','AUTO5_BOUWJAAR']]
Car5.columns = ['HHID','HH_cars', 'AUTO5', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car = pd.concat([Car1,Car2,Car3,Car4,Car5]) #Some HHs do not have cars or did not fill in any fueltypes
Car = Car.reset_index() #To avoid issues
Car.loc[Car['Car_year_MPN'] ==  9999, 'Car_year_MPN'] = np.nan #There are some 0 building years due to the .fillna(0) function, but these do not affect data as cardata is coupled based on building year from plate dataset directly

Car.loc[Car['AUTO1'] == 1, 'AUTO'] = 1
Car.loc[Car['AUTO2'] == 1, 'AUTO'] = 2
Car.loc[Car['AUTO3'] == 1, 'AUTO'] = 3
Car.loc[Car['AUTO4'] == 1, 'AUTO'] = 4
Car.loc[Car['AUTO5'] == 1, 'AUTO'] = 5

#Condense Fuel and Fuel_B into one category by identifying PHEVs and LPGs #Relevant for 2017?
Car.loc[(Car['Fueltype_MPN'] == 1) & (Car['Fuel_B'] == 5), 'Fueltype_MPN'] = 15 #Gasoline PHEV
Car.loc[(Car['Fueltype_MPN'] == 5) & (Car['Fuel_B'] == 1), 'Fueltype_MPN'] = 15 #Gasoline PHEV
Car.loc[(Car['Fueltype_MPN'] == 2) & (Car['Fuel_B'] == 5), 'Fueltype_MPN'] = 25 #Diesel PHEV
Car.loc[(Car['Fueltype_MPN'] == 5) & (Car['Fuel_B'] == 2), 'Fueltype_MPN'] = 25 #Diesel PHEV
Car.loc[(Car['Fuel_B'] == 3), 'Fueltype_MPN'] = 3 #LPG = LPG (even if first fuel type gasoline)

#Adding the hybrid vehicle class, power, official CO2 emissions, and (PH)EV MJ/vkm (TNO data) of each license plate
CarData = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/CarData3years.csv', index_col = 0)
Car = pd.merge(Car, CarData, left_on = 'Plate', right_on = 'Plate_CarData', how = 'left')
Car.loc[Car['Hybrid_type_RDW'] == 'NOVC-HEV', 'Fueltype_MPN'] = 4

Car.loc[(Car['Car_kg_RDW'].isna()) & (Car['Car_kg_empty_MPN'] > 0), 'Car_kg_RDW'] = Car['Car_kg_empty_MPN'] + 100 #Differences MPN and RDW data almost negligible, but sometimes MPN weight is known and RDW data is missing or vice versa

Car = Car[[ 'HHID','HH_cars', 'AUTO','Prob_carisusedmost','Fueltype_MPN', 'Car_design_MPN', 'Car_year_MPN', 'Car_ownership_MPN', 'Car_kg_RDW', 'Fuel_RDW_combined', 'CO2_RDW_combined', 'Nettmaxpower_RDW', 'CO2_RDW_WLTP_combined', 'Fuel_RDW_WLTP_combined', 'vkms_elec_TNO', 'MJ/vkm_EVs_TNO', 'MJ/vkm_fuel_PHEV_TNO', 'Fuel_norm_accordingtotravelcard', 'Fuel_real_travelcard', 'CO2_norm_accordingtotravelcard', 'CO2_real_travelcard']]


'''XXXXX Preprocessing the Weights XXXXX ''' 

Weights = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/Weights2018.csv', index_col = 0)
Weights = Weights[['HHID','WEEGHH2']] 

HH_weight = Weights.groupby('HHID', as_index=False)['WEEGHH2'].mean()
HH_weight.columns = ['HHID','HH_weight']
Weights = pd.merge(Weights, HH_weight, left_on='HHID', right_on='HHID', how = 'left')

Weights = Weights[['HHID','HH_weight']]
Weights = Weights.drop_duplicates(subset = 'HHID') 


'''XXXXX Combining the preprocessed questionnaires and making last refinements XXXX'''

MPN2018 = pd.merge(P, HH, left_on = 'HHID', right_on = 'HHID', how = 'left')
MPN2018 = pd.merge(MPN2018, Weights, left_on = 'HHID', right_on = 'HHID', how = 'left') 
MPN2018 = pd.merge(MPN2018, Car, left_on = 'HHID', right_on = 'HHID', how = 'left') #One entry per household car #Merging Car on MPN2018 instead of vice versa makes sure that households who do not own a car or have not provided any info on their car remain in the dataset.
MPN2018.loc[MPN2018['HH_cars'].isna(), 'HH_cars'] = MPN2018['HHAUTO_N'] #HHAUTO_N is car-count from hosuehold-questionnaire To avoid dropping all carless households

MPN2018 = MPN2018[(MPN2018['HH_VALID']==1)|(MPN2018['HH_VALID']==2)] #All HH members must have filled in the personal questionnaire. Otherwise, missing people become classified as #others (HHPERS remains reliable) and as non-highereducated. Moreover, the weight factor becomes less reliable

MPN2018.loc[MPN2018['AUTO'].isna(), 'AUTO'] = 0 #So carless households will also have a CARID
MPN2018['CARID'] = MPN2018['HHID']*10 + MPN2018['AUTO']
MPN2018 = MPN2018.drop_duplicates(subset = 'CARID')

MPN2018.loc[(MPN2018['HH_cars'] > 2), 'HH_cars'] = 2

#Assuming HHPERS includes the children under twelve 
MPN2018['HH_adults'] = MPN2018['HH_18to39'] + MPN2018['HH_40to59'] + MPN2018['HH_60plus'] #Ensures that missing household members are not included in the count
MPN2018['FracAdultMales'] = MPN2018['HH_AdultMales']/MPN2018['HH_adults']
MPN2018['FracAdultWorkers'] = MPN2018['HH_AdultWorkers']/MPN2018['HH_adults']
MPN2018['FracAdultHighedu'] =  MPN2018['HH_highereducated']/MPN2018['HH_adults'] #There are no teenagers with university degrees in the data


'''XXXXX Adding the incomes and PC6s (privacy-sensitive mergers) XXXX'''

#Adding the PC6 of each household
PC6 = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/HH_PC4_private2018.csv', index_col = 0)
PC6 = PC6[['HHID','WOONPC6']]
MPN2018combi = pd.merge(MPN2018, PC6, left_on = 'HHID', right_on = 'HHID', how = 'left')

#Adding data on each PC6 (like density)
PC6_vars = pd.read_csv('Datasets/PC6_2018_bewerkt_plusallmediumlargerhugecenters.csv', index_col = 0)
MPN2018combi['WOONPC6'] = MPN2018combi['WOONPC6'].str.upper() #Cause some postcodes are in small letters
MPN2018combi = pd.merge(MPN2018combi, PC6_vars, left_on = 'WOONPC6', right_on = 'PC6', how = 'left')

#Adding the household income
IncomePrivate = pd.read_csv(data_directory + os.sep + 'MPN2018_originaldata/Income_private2018.csv', index_col = 0)
IncomePrivate = IncomePrivate[['PERSID','HHBRUTOINK1_w5']]#Minor changes in category limits compared to 2019
MPN2018combi = pd.merge(MPN2018combi, IncomePrivate, left_on = 'PERSID', right_on = 'PERSID', how = 'left')
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 29, 'HHBRUTOINK1_w5'] = 28 #Unknown income category
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'].isna(), 'HHBRUTOINK1_w5'] = 28 #Unknown income category

#Income decategorization?
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 28, 'HHBRUTOINK1_w5'] = 14 #Unknown to average
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 1, 'HH_inc_real'] = (0 + 5000)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 2, 'HH_inc_real'] = (5000 + 6900)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 3, 'HH_inc_real'] = (6900 + 8700)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 4, 'HH_inc_real'] = (8700 + 10000)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 5, 'HH_inc_real'] = (10000 + 11800)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 6, 'HH_inc_real'] = (11800 + 13700)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 7, 'HH_inc_real'] = (13700 + 15600)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 8, 'HH_inc_real'] = (15600 + 16800)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 9, 'HH_inc_real'] = (16800 + 18700)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 10, 'HH_inc_real'] = (18700 + 21800)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 11, 'HH_inc_real'] = (21800 + 25500)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 12, 'HH_inc_real'] = (25500 + 28600)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 13, 'HH_inc_real'] = (28600 + 35500)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 14, 'HH_inc_real'] = (35500 + 42400)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 15, 'HH_inc_real'] = (42400 + 56100)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 16, 'HH_inc_real'] = (56100 + 71000)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 17, 'HH_inc_real'] = (71000 + 84700)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 18, 'HH_inc_real'] = (84700 + 113400)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 19, 'HH_inc_real'] = (113400 + 141400)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 20, 'HH_inc_real'] = (141400 + 169400)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 21, 'HH_inc_real'] = (169400 + 198100)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 22, 'HH_inc_real'] = (198100 + 225500)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 23, 'HH_inc_real'] = (225500 + 254100)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 24, 'HH_inc_real'] = (254100 + 282800)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 25, 'HH_inc_real'] = (282800 + 310800)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 26, 'HH_inc_real'] = (310800 + 339400)/2
MPN2018combi.loc[MPN2018combi['HHBRUTOINK1_w5'] == 27, 'HH_inc_real'] = 339400 + (339400-310800)/2

#Dropping useless columns, renaming columns, and then dropping all columns not to be used immediately
MPN2018combi = MPN2018combi[['HHID','CARID','JAAR','HH_VALID','HH_weight','FracAdultHighedu','FracAdultWorkers','N_KIND','HH_12to17','HH_adults','HH_18to39','HH_40to59','HH_60plus','HH_Dutch','FracAdultMales','HHBRUTOINK1_w5','HH_inc_real','OAD_PC5','AFS_SUPERM','AFS_OPRIT','AFS_TREINS','AFS_TRNOVS','km_center','km_mediumcenter','km_largercenter','km_hugecenter','ndvi_avg','landuse_idx5','f_4plus','bushalte1xpu','Tram_1000m','HHPARK1','HH_cars', 'Prob_carisusedmost','Car_ownership_MPN', 'Car_design_MPN','Car_year_MPN','Fueltype_MPN','Car_kg_RDW','Nettmaxpower_RDW','Fuel_RDW_combined', 'CO2_RDW_combined','vkms_elec_TNO','MJ/vkm_EVs_TNO','MJ/vkm_fuel_PHEV_TNO','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']]
MPN2018combi.columns = ['HHID','CARID','JAAR','HH_VALID','HH_weight','FracAdultHighedu','FracAdultWorkers','HH_under12','HH_12to17','HH_adults','HH_18to39','HH_40to59','HH_60plus','HH_Dutch','FracAdultMales','HH_inc','HH_inc_real','Density_PC5','km_super','km_highway','km_station','km_bigstation','km_center','km_mediumcenter','km_largercenter','km_hugecenter','NDVI','Landuse','f_4plus','km_bus','Tram_1000m','Parkingspot','HH_cars','Prob_carisusedmost', 'Car_ownership_MPN', 'Car_design_MPN','Car_year_MPN','Fueltype_MPN','Car_kg_RDW','Nettmaxpower_RDW','Fuel_RDW_combined', 'CO2_RDW_combined','vkms_elec_TNO','MJ/vkm_EVs_TNO','MJ/vkm_fuel_PHEV_TNO','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']

#Final selection based on privacy-sensitive data and saving
MPN2018combi = MPN2018combi[MPN2018combi['Density_PC5'] >= 0] 
MPN2018combi.loc[MPN2018combi['km_bus'].isna(), 'km_bus'] = MPN2018combi['km_bus'].mean() #The KiM seems to have not fixed the CBS subscript postcodes
MPN2018combi.to_csv(data_directory + os.sep + 'MPN_processed_data/MPN2018_combinedfile_1entrypercar.csv')


'''XXXXX Getting the descriptive statistics XXXXX '''
# # MPN2018independent = MPN2018combi[MPN2018combi['HH_cars'] > 0]
# MPN2018independent = MPN2018combi[['GESLACHT','OPLEIDING','KLEEFT2','WERKSITUATIE_MEEST_w5','Density_PC5','km_station','km_center','NDVI','Landuse','km_bus','ParkingSpot','HH_inc','HH_undertwelve','HH_workers','HH_students','HH_other','HH_highedu']]
# MPN2018independent.columns = ['GESLACHT','OPLEIDING','KLEEFT2','WERKSITUATIE_MEEST_w5','Address density PC5 (addresses/km2)','Distance to train station (km)','Distance to center (km)','Green space (NDVI)','Landuse mix entropy (index)','Distance to bus stop (km)','Parking','Income (ordinal)','Young children (# people)','Workers (# people)','Students (# people)','Other (# people)','Higher educated (# people)']
# MPN2018description = MPN2018independent.describe()
# MPN2018description = MPN2018description.round(2)
# MPN2018description = MPN2018description.T
# MPN2018description = MPN2018description[['min','25%','50%','mean','75%','max','std']]
# print(MPN2018description.to_latex())
