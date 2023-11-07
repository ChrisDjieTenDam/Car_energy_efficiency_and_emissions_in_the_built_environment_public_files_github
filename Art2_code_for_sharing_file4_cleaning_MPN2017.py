# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:06:12 2021 by Chris ten Dam.
This code has been written for an academic study titled "Driving energy and emissions: the built environment influence on car efficiency".

It cleans the data from the 2017 Netherlands Mobility Panel.
There are multiple datasets: one for each survey.

After cleaning and combining the relevant data, the privacy-sensitive data on household incomes and residential environment is added.
Finally, the identifyers (license plates and PC6-addresses) are dropped, along with all other columns that will not be used immediately. 
"""

import os

data_directory = r"Q:\research-driving-energy"

import numpy as np
import pandas as pd


'''XXXXX Preprocessing the Personal data questionnaires XXXXX '''

P = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/P2017.csv', usecols = ['HHID','HH_VALID','PERSID','GESLACHT','HERKOMST_w5', 'WERKSITUATIE_MEEST_w5','WERKURENRECENT_CONTINU', 'OPLEIDING','KLEEFT2'])

P['OPLEIDING'].replace(to_replace = 8, value = 7, inplace = True) #Higher education (HBO & WO) according to CBS classification
P.loc[P['OPLEIDING'] == 7, 'Highereducated'] = 1 #Everybody with higher education (HBO & WO)
HH_highereducated = P.groupby('HHID', as_index=False)['Highereducated'].sum()
HH_highereducated.columns = ['HHID','HH_highereducated']
P = pd.merge(P, HH_highereducated, left_on='HHID', right_on='HHID', how = 'left')

###Societal position categories
P['SP'] = P['WERKSITUATIE_MEEST_w5']
for other in [4,5,8,9,10,99]: #Keep to prevent part time workers to be misclassified as fulltime workers
    P.loc[P['SP'] == other, 'SP'] = 11
P.loc[(P['WERKURENRECENT_CONTINU'] < 24) & (P['SP'] != 6) & (P['SP'] != 7), 'SP'] = 11 #Working less than three days a week to other

P.loc[(P['SP'] == 1)|(P['SP'] == 2)|(P['SP'] == 3), 'SP'] = 1#3days/week self-employed, employed by private organization, or employed by government
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

HH = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/HH2017.csv', usecols = ['HHID','JAAR','HHPERS','N_KIND','HHAUTO_N','HHPARK1','bushalte1xpu','tramhalte']) 

HH['bushalte1xpu'] = HH['bushalte1xpu']/1000 #meters to kilometers
HH['Tram_1000m'] = 0
HH.loc[(HH['tramhalte'] <= 1000), 'Tram_1000m'] = 1 #Because just distance to tram becomes a proxy for distance to Randstad

HH.loc[HH['HHPARK1'] == 99, 'HHPARK1'] = 0 #"no household questionnaire, no imputation from 2017 possible"

HH = HH[['HHID','JAAR','HHAUTO_N','HHPERS','N_KIND','HHPARK1','bushalte1xpu','Tram_1000m']]


'''XXXXX Preprocessing the Car data questionnaires  and adding (privacy_sesitive) plates XXXXX ''' #

Car = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/Car2017.csv')

#Only keep cars which are known to exist #There some cars with "gasoline" or "other" fuel type with AUTO1 = 99 or 0, this code assumes that these are data errors
for auto in ['AUTO1','AUTO2','AUTO3','AUTO4','AUTO5']:
    Car[auto].replace(to_replace = 99, value = np.nan, inplace = True)
    Car[auto].replace(to_replace = 0, value = np.nan, inplace = True) 

#Find the highest distance category (vkm/year) for each household #AUTO_1KM 2017 matches categories in 2018 and 2019 (same limits)
for km in ['AUTO1_KM','AUTO2_KM','AUTO3_KM','AUTO4_KM','AUTO5_KM']: #Replace with ones instead of nans cause otherwise, people with only 1 car with a 97 or 6 value get ignored (inter alia reducing the number of EVs)
    Car.loc[Car[km] >= 6, km] = np.nan #"don't know/won't say", though it is arguable that these cars are actually used a lot (57 cases in AUTO1_KM, 16 in AUTO2_KM, and 3 in AUTO3_KM)
    Car.loc[Car[km].isna(), km] = Car[km].mean()

#Adding the car license plates and building years #PRIVACY-SENSITIVE MERGER
Plate = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/Plate_private2017.csv', index_col = 0)
Car = pd.merge(Car, Plate, left_on = 'HHID', right_on = 'HHID', how = 'left')
for plate in ['KENTEKEN1','KENTEKEN2','KENTEKEN3','KENTEKEN4','KENTEKEN5']:
    Car[plate] = Car[plate].str.upper() #Cause some plates are in small letters

Vars = pd.read_csv('Mostusedcar_model_variables.csv')

Car['AUTO1_BOUWJAAR_regression'] = Car['AUTO1_BOUWJAAR'] #KAUTO_BOUWJAAR is always nan in 2017
Car['AUTO2_BOUWJAAR_regression'] = Car['AUTO2_BOUWJAAR']
Car['AUTO3_BOUWJAAR_regression'] = Car['AUTO3_BOUWJAAR']
Car['AUTO4_BOUWJAAR_regression'] = Car['AUTO4_BOUWJAAR']
Car['AUTO5_BOUWJAAR_regression'] = Car['AUTO5_BOUWJAAR']

for jaar in ['AUTO1_BOUWJAAR_regression','AUTO2_BOUWJAAR_regression','AUTO3_BOUWJAAR_regression','AUTO4_BOUWJAAR_regression','AUTO5_BOUWJAAR_regression']:
    Car.loc[Car[jaar] == 9999, jaar] = np.nan 
    Car.loc[(Car[jaar] > 1990) & (Car[jaar] <= 2000), jaar] = 1995
    Car.loc[(Car[jaar] <= 1990), jaar] = 1985 #Recreate categorization of KAUTO_BOUWJAAR so that old cars do not mess up the regression outcomes

Car['age1'] = 2017 - Car['AUTO1_BOUWJAAR_regression'] #AUTO_BOUWJAAR is always nan for car 4
Car['age2'] = 2017 - Car['AUTO2_BOUWJAAR_regression'] 
Car['age3'] = 2017 - Car['AUTO3_BOUWJAAR_regression']

for leeftijd in ['age1','age2','age3']:
     Car.loc[Car[leeftijd].isna(), leeftijd] = Car[leeftijd].mean() #So that nonvalid cars do not get assigned a zero age (and thus a very high probability)
Car['age4'] = 9.4 #The mean age of the 4th car (as well as all cars) in 2019
Car['age5'] = 9.4 #The mean age of the 4th car (as well as all cars) in 2019

Car = Car.fillna(0)

Car['logit_1'] = Vars['KM_category'].item()*Car['AUTO1_KM'] + Vars['Age'].item()*Car['age1'] 
Car['logit_2'] = Vars['KM_category'].item()*Car['AUTO2_KM'] + Vars['Age'].item()*Car['age2'] 
Car['logit_3'] = Vars['KM_category'].item()*Car['AUTO3_KM'] + Vars['Age'].item()*Car['age3'] 
Car['logit_4'] = Vars['KM_category'].item()*Car['AUTO4_KM'] + Vars['Age'].item()*Car['age4'] 
Car['logit_5'] = Vars['KM_category'].item()*Car['AUTO5_KM'] + Vars['Age'].item()*Car['age5'] 

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
Car1 = Car1[['HHID','HHAUTO_N', 'AUTO1', 'AUTO1_BRANDSTOF_A', 'AUTO1_BRANDSTOF_B', 'Prob_car1isusedmost', 'AUTO1_AANSCHAF', 'AUTO1_GEWLEEG','AUTO1_INRICHT','KENTEKEN1','AUTO1_BOUWJAAR']]
Car1.columns = ['HHID','HH_cars', 'AUTO1', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car2 = Car[Car['AUTO2'] == 1]
Car2 = Car2[['HHID','HHAUTO_N', 'AUTO2', 'AUTO2_BRANDSTOF_A', 'AUTO2_BRANDSTOF_B', 'Prob_car2isusedmost', 'AUTO2_AANSCHAF', 'AUTO2_GEWLEEG','AUTO2_INRICHT','KENTEKEN2','AUTO2_BOUWJAAR']]
Car2.columns = ['HHID','HH_cars', 'AUTO2', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car3 = Car[Car['AUTO3'] == 1]
Car3 = Car3[['HHID','HHAUTO_N', 'AUTO3', 'AUTO3_BRANDSTOF_A', 'AUTO3_BRANDSTOF_B', 'Prob_car3isusedmost', 'AUTO3_AANSCHAF', 'AUTO3_GEWLEEG','AUTO3_INRICHT','KENTEKEN3','AUTO3_BOUWJAAR']]
Car3.columns = ['HHID','HH_cars', 'AUTO3', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car4 = Car[Car['AUTO4'] == 1] #17 cases. 0 cases of 5th cars being used most
Car4 = Car4[['HHID','HHAUTO_N', 'AUTO4', 'AUTO4_BRANDSTOF_A', 'AUTO4_BRANDSTOF_B', 'Prob_car4isusedmost', 'AUTO4_AANSCHAF', 'AUTO4_GEWLEEG','AUTO4_INRICHT','KENTEKEN4','AUTO4_BOUWJAAR']]
Car4.columns = ['HHID','HH_cars', 'AUTO4', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car5 = Car[Car['AUTO5'] == 1] #2 cases
Car5 = Car5[['HHID','HHAUTO_N', 'AUTO5', 'AUTO5_BRANDSTOF_A', 'AUTO5_BRANDSTOF_B', 'Prob_car5isusedmost', 'AUTO5_AANSCHAF', 'AUTO5_GEWLEEG','AUTO5_INRICHT','KENTEKEN5','AUTO5_BOUWJAAR']]
Car5.columns = ['HHID','HH_cars', 'AUTO5', 'Fueltype_MPN', 'Fuel_B', 'Prob_carisusedmost', 'Car_ownership_MPN', 'Car_kg_empty_MPN','Car_design_MPN','Plate','Car_year_MPN']

Car = pd.concat([Car1,Car2,Car3,Car4,Car5]) #Some HHs do not have cars or did not fill in any fueltypes
Car = Car.reset_index() #To avoid issues
Car.loc[Car['Car_year_MPN'] ==  9999, 'Car_year_MPN'] = np.nan 

Car.loc[Car['AUTO1'] == 1, 'AUTO'] = 1 #To get seperate CARIDs below
Car.loc[Car['AUTO2'] == 1, 'AUTO'] = 2
Car.loc[Car['AUTO3'] == 1, 'AUTO'] = 3
Car.loc[Car['AUTO4'] == 1, 'AUTO'] = 4
Car.loc[Car['AUTO5'] == 1, 'AUTO'] = 5

#Car characteristics of most driven car. Keep newest cars if more than one in same distance category.
#Fuel type cat. 2017 almost equal to 2018/2019. Fuel_A == 3 now includes with petrol and Fuel_A == 5 now means fully electric (BEV)
Car.loc[Car['Car_kg_empty_MPN'] < 5, 'Car_kg_empty_MPN'] = Car['Car_kg_empty_MPN']*1000 #weights of more than 1000kg are in tons

#Condense Fuel and Fuel_B into one category by identifying PHEVs and LPGs 
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

Car = Car[[ 'HHID', 'HH_cars','AUTO', 'Prob_carisusedmost', 'Fueltype_MPN', 'Car_design_MPN', 'Car_year_MPN', 'Car_ownership_MPN', 'Car_kg_RDW', 'Fuel_RDW_combined', 'CO2_RDW_combined', 'Nettmaxpower_RDW', 'CO2_RDW_WLTP_combined', 'Fuel_RDW_WLTP_combined', 'vkms_elec_TNO', 'MJ/vkm_EVs_TNO', 'MJ/vkm_fuel_PHEV_TNO', 'Fuel_norm_accordingtotravelcard', 'Fuel_real_travelcard', 'CO2_norm_accordingtotravelcard', 'CO2_real_travelcard']]


'''XXXXX Preprocessing the Weights XXXXX ''' 

Weights = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/Weights2017.csv', index_col = 0)
Weights = Weights[['HHID','WEEGHH2']] 

HH_weight = Weights.groupby('HHID', as_index=False)['WEEGHH2'].mean()
HH_weight.columns = ['HHID','HH_weight']
Weights = pd.merge(Weights, HH_weight, left_on='HHID', right_on='HHID', how = 'left')

Weights = Weights[['HHID','HH_weight']]
Weights = Weights.drop_duplicates(subset = 'HHID') #Creates a lot of duplicates in the merger otherwise


'''XXXXX Combining the preprocessed questionnaires and making last refinements XXXX'''

MPN2017 = pd.merge(P, HH, left_on = 'HHID', right_on = 'HHID', how = 'left')
MPN2017 = pd.merge(MPN2017, Weights, left_on = 'HHID', right_on = 'HHID', how = 'left')
MPN2017 = pd.merge(MPN2017, Car, left_on = 'HHID', right_on = 'HHID', how = 'left') #One entry per household car
MPN2017.loc[MPN2017['HH_cars'].isna(), 'HH_cars'] = MPN2017['HHAUTO_N'] #HHAUTO_N is car-count from household-questionnaire To avoid dropping all carless households
MPN2017 = MPN2017[(MPN2017['HH_VALID']==1)|(MPN2017['HH_VALID']==2)] #All HH members must have filled in the personal questionnaire. Otherwise, missing people become classified as #others (HHPERS remains reliable) and as non-highereducated. Moreover, the weight factor becomes less reliable

MPN2017.loc[MPN2017['AUTO'].isna(), 'AUTO'] = 0 #So carless households will also have a CARID
MPN2017['CARID'] = MPN2017['HHID']*10 + MPN2017['AUTO']
MPN2017 = MPN2017.drop_duplicates(subset = 'CARID') 

MPN2017.loc[(MPN2017['HH_cars'] > 2), 'HH_cars'] = 2

#Assuming HHPERS includes the children under twelve #Singling out non-working and non-studying HH-members to reduce multicolinearity (by replacing HHPERS)
MPN2017['HH_adults'] = MPN2017['HH_18to39'] + MPN2017['HH_40to59'] + MPN2017['HH_60plus'] #Ensures that missing household members are not included in the count
MPN2017['FracAdultMales'] = MPN2017['HH_AdultMales']/MPN2017['HH_adults']
MPN2017['FracAdultWorkers'] = MPN2017['HH_AdultWorkers']/MPN2017['HH_adults']
MPN2017['FracAdultHighedu'] =  MPN2017['HH_highereducated']/MPN2017['HH_adults']


'''XXXXX Adding the incomes and PC6s (privacy-sensitive mergers) XXXX'''

#Adding the PC6 of each household
PC6 = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/HH_PC4_private2017.csv', index_col = 0)
PC6 = PC6[['HHID','WOONPC6']]
MPN2017combi = pd.merge(MPN2017, PC6, left_on = 'HHID', right_on = 'HHID', how = 'left')

#Adding data on each PC6 (like density) 
PC6_vars = pd.read_csv('Datasets/PC6_2018_bewerkt_plusallmediumlargerhugecenters.csv', index_col = 0)
MPN2017combi['WOONPC6'] = MPN2017combi['WOONPC6'].str.upper() #Cause some postcodes are in small letters?
MPN2017combi = pd.merge(MPN2017combi, PC6_vars, left_on = 'WOONPC6', right_on = 'PC6', how = 'left')

#Adding the household income
IncomePrivate = pd.read_csv(data_directory + os.sep + 'MPN2017_originaldata/Income_private2017.csv', index_col = 0)
IncomePrivate = IncomePrivate[['PERSID','HHBRUTOINK1_w5']]#income cat. slightly different from both 2018 and 2017
MPN2017combi = pd.merge(MPN2017combi, IncomePrivate, left_on = 'PERSID', right_on = 'PERSID', how = 'left')
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 29, 'HHBRUTOINK1_w5'] = 28 #Unknown income category
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'].isna(), 'HHBRUTOINK1_w5'] = 28 #Unknown income category

#Income decategorization?
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 28, 'HHBRUTOINK1_w5'] = 14 #Unknown to average
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 1, 'HH_inc_real'] = (0 + 4700)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 2, 'HH_inc_real'] = (4700 + 6500)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 3, 'HH_inc_real'] = (6500 + 8200)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 4, 'HH_inc_real'] = (8200 + 9400)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 5, 'HH_inc_real'] = (9400 + 11200)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 6, 'HH_inc_real'] = (11200 + 12900)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 7, 'HH_inc_real'] = (12900 + 14700)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 8, 'HH_inc_real'] = (14700 + 15900)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 9, 'HH_inc_real'] = (15900 + 17600)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 10, 'HH_inc_real'] = (17600 + 20600)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 11, 'HH_inc_real'] = (20600 + 24100)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 12, 'HH_inc_real'] = (24100 + 27000)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 13, 'HH_inc_real'] = (27000 + 33500)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 14, 'HH_inc_real'] = (33500 + 40000)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 15, 'HH_inc_real'] = (40000 + 52900)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 16, 'HH_inc_real'] = (52900 + 67000)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 17, 'HH_inc_real'] = (67000 + 79900)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 18, 'HH_inc_real'] = (79900 + 107000)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 19, 'HH_inc_real'] = (107000 + 133400)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 20, 'HH_inc_real'] = (133400 + 159900)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 21, 'HH_inc_real'] = (159900 + 186900)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 22, 'HH_inc_real'] = (186900 + 212800)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 23, 'HH_inc_real'] = (212800 + 239800)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 24, 'HH_inc_real'] = (239800 + 266800)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 25, 'HH_inc_real'] = (266800 + 293300)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 26, 'HH_inc_real'] = (293300 + 320300)/2
MPN2017combi.loc[MPN2017combi['HHBRUTOINK1_w5'] == 27, 'HH_inc_real'] = 320300 + (320300-293300)/2

#Dropping useless columns, renaming columns, and then dropping all columns not to be used immediately
MPN2017combi = MPN2017combi[['HHID','CARID','JAAR','HH_VALID','HH_weight','FracAdultHighedu','FracAdultWorkers','N_KIND','HH_12to17','HH_adults','HH_18to39','HH_40to59','HH_60plus','HH_Dutch','FracAdultMales','HHBRUTOINK1_w5','HH_inc_real','OAD_PC5','AFS_SUPERM','AFS_OPRIT','AFS_TREINS','AFS_TRNOVS','km_center','km_mediumcenter','km_largercenter','km_hugecenter','ndvi_avg','landuse_idx5','f_4plus','bushalte1xpu','Tram_1000m','HHPARK1','HH_cars','Prob_carisusedmost','Car_ownership_MPN','Car_design_MPN','Car_year_MPN','Fueltype_MPN','Car_kg_RDW','Nettmaxpower_RDW','Fuel_RDW_combined','CO2_RDW_combined','vkms_elec_TNO','MJ/vkm_EVs_TNO','MJ/vkm_fuel_PHEV_TNO','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']]
MPN2017combi.columns = ['HHID','CARID','JAAR','HH_VALID','HH_weight','FracAdultHighedu','FracAdultWorkers','HH_under12','HH_12to17','HH_adults','HH_18to39','HH_40to59','HH_60plus','HH_Dutch','FracAdultMales','HH_inc','HH_inc_real','Density_PC5','km_super','km_highway','km_station','km_bigstation','km_center','km_mediumcenter','km_largercenter','km_hugecenter','NDVI','Landuse','f_4plus','km_bus','Tram_1000m','Parkingspot','HH_cars','Prob_carisusedmost','Car_ownership_MPN','Car_design_MPN','Car_year_MPN','Fueltype_MPN','Car_kg_RDW','Nettmaxpower_RDW','Fuel_RDW_combined','CO2_RDW_combined','vkms_elec_TNO','MJ/vkm_EVs_TNO','MJ/vkm_fuel_PHEV_TNO','Fuel_norm_accordingtotravelcard','Fuel_real_travelcard','CO2_norm_accordingtotravelcard','CO2_real_travelcard']

#Final selection based on privacy-sensitive data and saving
MPN2017combi = MPN2017combi[MPN2017combi['Density_PC5'] >= 0] 
MPN2017combi.loc[MPN2017combi['km_bus'].isna(), 'km_bus'] = MPN2017combi['km_bus'].mean() #The KiM seems to have not fixed the CBS subscript postcodes
MPN2017combi.to_csv(data_directory + os.sep + 'MPN_processed_data/MPN2017_combinedfile_1entrypercar.csv')


'''XXXXX Getting the descriptive statistics XXXXX '''
# # MPN2017independent = MPN2017combi[MPN2017combi['HH_cars'] > 0]
# MPN2017independent = MPN2017combi[['GESLACHT','OPLEIDING','KLEEFT2','WERKSITUATIE_MEEST_w5','Density_PC5','km_station','km_center','NDVI','Landuse','km_bus','ParkingSpot','HH_inc','HH_undertwelve','HH_workers','HH_students','HH_other','HH_highedu']]
# MPN2017independent.columns = ['GESLACHT','OPLEIDING','KLEEFT2','WERKSITUATIE_MEEST_w5','Address density PC5 (addresses/km2)','Distance to train station (km)','Distance to center (km)','Green space (NDVI)','Landuse mix entropy (index)','Distance to bus stop (km)','Parking','Income (ordinal)','Young children (# people)','Workers (# people)','Students (# people)','Other (# people)','Higher educated (# people)']
# MPN2017description = MPN2017independent.describe()
# MPN2017description = MPN2017description.round(2)
# MPN2017description = MPN2017description.T
# MPN2017description = MPN2017description[['min','25%','50%','mean','75%','max','std']]
# print(MPN2017description.to_latex())
