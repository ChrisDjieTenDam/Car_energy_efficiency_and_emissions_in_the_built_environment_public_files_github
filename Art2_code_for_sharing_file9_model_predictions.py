# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:29:48 2023 by Chris ten Dam.
This code has been written for an academic study titled "Driving energy and emissions: the built environment influence on car efficiency".

It can be used to make the predictions for sociodemographic and built environmnet profiles (e.g. a student in Amsterdam).
"""

'''XXXXX Preparing the data XXXXX'''

import os
data_directory = r"Q:\research-driving-energy"

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, linear_model

MPN = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/MPN3years_1entrypercar.csv', index_col = 0) #The main file 
MPN.loc[MPN['HH_cars'] > 1,'HH_weight'] = MPN['HH_weight']*MPN['Prob_carisusedmost'] #Prob = 1 for households with one (or no) car
MPN = MPN.astype(float) 
MPN['HH_weight'] = MPN['HH_weight']*1.1316046025500308

train = MPN[['HHID','HH_cars','HH_weight','Type','CarValid','HH_under12','HH_12to17','HH_18to39','HH_40to59','HH_60plus','FracAdultMales','FracAdultWorkers','FracAdultHighedu','HH_inc_real','Density_PC5','km_center','km_mediumcenter','km_largercenter','km_hugecenter','Landuse','NDVI','Parkingspot','km_super','km_station','km_bigstation','km_bus']]
train = train.dropna()

train['HH_under18'] = train['HH_under12'] + train['HH_12to17']
train['HH_18to59'] = train['HH_18to39'] + train['HH_40to59']

#Going from strange income classes to the mean real income of each class
train['HH_inc_low'] = 0
train.loc[train['HH_inc_real'] < 20000, 'HH_inc_low'] = 1 
train['HH_inc_middle'] = 0
train.loc[(train['HH_inc_real'] >= 40000) & (train['HH_inc_real'] < 60000), 'HH_inc_middle'] = 1 
train['HH_inc_middlehigh'] = 0
train.loc[(train['HH_inc_real'] >= 60000) & (train['HH_inc_real'] < 120000), 'HH_inc_middlehigh'] = 1 
train['HH_inc_high'] = 0
train.loc[(train['HH_inc_real'] >= 120000), 'HH_inc_high'] = 1 

train.loc[train['CarValid'] == 0, 'Type'] = 13 #No valid car means standard-fueled midlight car. These people will be excluded through the CarValid indicator though. 

train['km_station'] = np.log(train['km_station'])
train['km_bigstation'] = np.log(train['km_bigstation'])

#Scaling
xvars = ['HH_under18','HH_18to59','HH_18to39','HH_40to59','HH_60plus','FracAdultMales','FracAdultWorkers','FracAdultHighedu','HH_inc_low', 'HH_inc_middle','HH_inc_middlehigh', 'HH_inc_high','Density_PC5','km_center','km_mediumcenter','km_largercenter','km_hugecenter','Landuse','NDVI','Parkingspot','km_super','km_station','km_bigstation','km_bus']
train_HHIDunique = train.drop_duplicates(subset = 'HHID') 

standard_scaler = preprocessing.StandardScaler() 
standard_scaler.fit(train_HHIDunique[xvars]) #Scale based on 4316 households to match descriptive statistics


'''XXXXX To get predictions for the original training data (comment out below code for specific profiles) XXXXX'''
##For comparison purposes

# train_HHIDunique[xvars] = standard_scaler.transform(train_HHIDunique[xvars])
# train_HHIDunique = train_HHIDunique[['HH_cars','HH_weight','Type','CarValid','HH_under18','HH_18to39','HH_40to59','HH_60plus','FracAdultMales','FracAdultWorkers','FracAdultHighedu','HH_inc_low', 'HH_inc_middle','HH_inc_middlehigh', 'HH_inc_high','Density_PC5','km_largercenter','km_hugecenter','Landuse','NDVI','Parkingspot','km_station','km_bigstation','km_bus']]
# profiles = train_HHIDunique #Compute the vehicle energy use of the 4316 households. The probability of owning no car, one car, or multiple cars is taken care of by the Latent Class model. 


'''XXXXX To get predictions for specific profiles: (comment out above code for original training data) XXXXX'''

profiles = pd.DataFrame(columns = train.columns) 

##Only use the profile that you want to get predictions for (the high income family in Amsterdam). Comment out the other lines.
# profiles.loc['Student_1012NH'] = [1,0,1,11,1,0,0,1,0,0,0,0,1,0,9217,0.366,0.366,0.366,0.366,0.872,0.154,0,0.5,0.7,0.7,0.088,0,1,1,0,0,0] #Amsterdam
# profiles.loc['Student_5384AK'] = [1,0,1,11,1,0,0,1,0,0,0,0,1,0,1196,4.362,4.362,16.056,48.702,0.622,0.450,1,0.2,3.9,18.5,0.123,0,1,1,0,0,0] #Heesch
# profiles.loc['Student_8741KB'] = [1,0,1,11,1,0,0,1,0,0,0,0,1,0,17,  7.922,7.922,21.135,90.170,0.066,0.711,1,3.8,9.9,25.9,0.253,0,1,1,0,0,0] #Friesland farmland
# profiles.loc['Youngfamily_1012NH'] = [1,1,1,12,1,2,0,2,0,0,0.5,1,0,50000,9217,0.366,0.366,0.366,0.366,0.872,0.154,0,0.5,0.7,0.7,0.088,2,2,0,1,0,0]
# profiles.loc['Youngfamily_5384AK'] = [1,1,1,12,1,2,0,2,0,0,0.5,1,0,50000,1196,4.362,4.362,16.056,48.702,0.622,0.450,1,0.2,3.9,18.5,0.123,2,2,0,1,0,0]
# profiles.loc['Youngfamily_8741KB'] = [1,1,1,12,1,2,0,2,0,0,0.5,1,0,50000,17,7.922,7.922,21.135,90.170,0.066,0.711,1,3.8,9.9,25.9,0.253,2,2,0,1,0,0]
profiles.loc['Richfamily_1012NH'] = [1,1,1,12,1,2,0,0,2,0,0.5,1,0,150000,9217,0.366,0.366,0.366,0.366,0.872,0.154,0,0.5,0.7,0.7,0.088,2,2,0,0,0,1]
# profiles.loc['Richfamily_5384AK'] = [1,1,1,12,1,2,0,0,2,0,0.5,1,0,150000,1196,4.362,4.362,16.056,48.702,0.622,0.450,1,0.2,3.9,18.5,0.123,2,2,0,0,0,1]
# profiles.loc['Richfamily_8741KB'] = [1,1,1,12,1,2,0,0,2,0,0.5,1,0,150000,17,7.922,7.922,21.135,90.170,0.066,0.711,1,3.8,9.9,25.9,0.253,2,2,0,0,0,1]

#The distances to stations must be logtransformed
profiles['km_station'] = np.log(profiles['km_station'])
profiles['km_bigstation'] = np.log(profiles['km_bigstation'])

#The data must be standardized based on the mean and standard deviation of the 4316 included households
profiles[xvars] = standard_scaler.transform(profiles[xvars])
profiles = profiles[['HH_weight','HH_under18','HH_18to39','HH_40to59','HH_60plus','FracAdultMales','FracAdultWorkers','FracAdultHighedu','HH_inc_low', 'HH_inc_middle','HH_inc_middlehigh', 'HH_inc_high','Density_PC5','km_largercenter','km_hugecenter','Landuse','NDVI','Parkingspot','km_station','km_bigstation','km_bus']]

#To get 100k draws
profiles = profiles.append([profiles]*99999,ignore_index=True) 


'''XXXXX Computing the probabilities XXXXX'''

BioParameters = pd.read_excel('20231020_BioParameters_HHIDuniqueScaler.xlsx') 

#The utility of each car ownership class
U_carless = 0
U_onecar = BioParameters['OwnscarConstant'].item() + BioParameters['Ownscar_FracAdultWorkers'].item()*profiles['FracAdultWorkers'] + BioParameters['Ownscar_HH_00under18'].item()*profiles['HH_under18'] + BioParameters['Ownscar_HH_18to39'].item()*profiles['HH_18to39'] + BioParameters['Ownscar_HH_40to59'].item()*profiles['HH_40to59'] + BioParameters['Ownscar_HH_60plus'].item()*profiles['HH_60plus'] + BioParameters['Ownscar_HH_inc_0low'].item()*profiles['HH_inc_low'] + BioParameters['Ownscar_HH_inc_1middle'].item()*profiles['HH_inc_middle']  + BioParameters['Ownscar_HH_inc_2middlehigh'].item()*profiles['HH_inc_middlehigh'] + BioParameters['Ownscar_HH_inc_3high'].item()*profiles['HH_inc_high'] + BioParameters['Ownscar_Density_PC5'].item()*profiles['Density_PC5'] + BioParameters['Ownscar_km_hugecenter'].item()*profiles['km_hugecenter'] + BioParameters['Ownscar_km_station'].item()*profiles['km_station'] + BioParameters['Ownscar_km_bigstation'].item()*profiles['km_bigstation'] + BioParameters['Ownscar_km_bus'].item()*profiles['km_bus'] + BioParameters['Ownscar_Parkingspot'].item()*profiles['Parkingspot']
U_twocar = BioParameters['TwocarConstant'].item() + BioParameters['Twocar_FracAdultMales'].item()*profiles['FracAdultMales'] + BioParameters['Twocar_FracAdultWorkers'].item()*profiles['FracAdultWorkers'] + BioParameters['Twocar_HH_00under18'].item()*profiles['HH_under18'] + BioParameters['Twocar_HH_18to39'].item()*profiles['HH_18to39'] + BioParameters['Twocar_HH_40to59'].item()*profiles['HH_40to59'] + BioParameters['Twocar_HH_60plus'].item()*profiles['HH_60plus'] + BioParameters['Twocar_HH_inc_0low'].item()*profiles['HH_inc_low'] + BioParameters['Twocar_HH_inc_1middle'].item()*profiles['HH_inc_middle'] + BioParameters['Twocar_HH_inc_2middlehigh'].item()*profiles['HH_inc_middlehigh'] + BioParameters['Twocar_HH_inc_3high'].item()*profiles['HH_inc_high'] + BioParameters['Twocar_Density_PC5'].item()*profiles['Density_PC5'] + BioParameters['Twocar_km_hugecenter'].item()*profiles['km_hugecenter'] + BioParameters['Twocar_km_station'].item()*profiles['km_station'] + BioParameters['Twocar_km_bigstation'].item()*profiles['km_bigstation'] + BioParameters['Twocar_km_bus'].item()*profiles['km_bus'] + BioParameters['Twocar_Parkingspot'].item()*profiles['Parkingspot']

#The probability of belonging to each car ownership class
profiles['Pcarless'] = np.e**(U_carless)/(np.e**(U_carless) + np.e**(U_onecar) + np.e**(U_twocar))
profiles['Ponecar'] = np.e**(U_onecar)/(np.e**(U_carless) + np.e**(U_onecar) + np.e**(U_twocar))
profiles['Ptwocar'] = np.e**(U_twocar)/(np.e**(U_carless) + np.e**(U_onecar) + np.e**(U_twocar))

#The utility of each fuel type and weight category for the one-car class
U_std = BioParameters['Std_FracAdultMales'].item()*profiles['FracAdultMales'] + BioParameters['Std_FracAdultHighedu'].item()*profiles['FracAdultHighedu'] + BioParameters['Std_HH_18to39'].item()*profiles['HH_18to39'] + BioParameters['Std_HH_inc_0low'].item()*profiles['HH_inc_low'] + BioParameters['Std_km_largercenter'].item()*profiles['km_largercenter'] + BioParameters['Std_km_hugecenter'].item()*profiles['km_hugecenter']
U_diesel = BioParameters['Diesel_FracAdultWorkers'].item()*profiles['FracAdultWorkers'] + BioParameters['Diesel_HH_18to39'].item()*profiles['HH_18to39'] + BioParameters['Diesel_HH_40to59'].item()*profiles['HH_40to59'] + BioParameters['Diesel_km_largercenter'].item()*profiles['km_largercenter'] + BioParameters['Diesel_Landuse'].item()*profiles['Landuse'] + BioParameters['Diesel_Parkingspot'].item()*profiles['Parkingspot']
U_light = BioParameters['Light_FracAdultMales'].item()*profiles['FracAdultMales'] + BioParameters['Light_HH_00under18'].item()*profiles['HH_under18'] + BioParameters['Light_HH_40to59'].item()*profiles['HH_40to59'] + BioParameters['Light_HH_60plus'].item()*profiles['HH_60plus'] + BioParameters['Light_HH_inc_1middle'].item()*profiles['HH_inc_middle'] + BioParameters['Light_HH_inc_2middlehigh'].item()*profiles['HH_inc_middlehigh'] + BioParameters['Light_HH_inc_3high'].item()*profiles['HH_inc_high'] + BioParameters['Light_km_hugecenter'].item()*profiles['km_hugecenter'] + BioParameters['Light_NDVI'].item()*profiles['NDVI'] + BioParameters['Light_Parkingspot'].item()*profiles['Parkingspot']
U_midlight = BioParameters['Midlight_FracAdultWorkers'].item()*profiles['FracAdultWorkers'] + BioParameters['Midlight_HH_00under18'].item()*profiles['HH_under18'] + BioParameters['Midlight_HH_60plus'].item()*profiles['HH_60plus'] + BioParameters['Midlight_HH_inc_3high'].item()*profiles['HH_inc_high'] + BioParameters['Midlight_km_hugecenter'].item()*profiles['km_hugecenter']
U_midheavy = BioParameters['Midheavy_FracAdultWorkers'].item()*profiles['FracAdultWorkers'] + BioParameters['Midheavy_HH_40to59'].item()*profiles['HH_40to59'] + BioParameters['Midheavy_HH_inc_2middlehigh'].item()*profiles['HH_inc_middlehigh'] + BioParameters['Midheavy_NDVI'].item()*profiles['NDVI']
U_heavy = BioParameters['Heavy_FracAdultMales'].item()*profiles['FracAdultMales'] + BioParameters['Heavy_FracAdultHighedu'].item()*profiles['FracAdultHighedu'] + BioParameters['Heavy_HH_00under18'].item()*profiles['HH_under18'] + BioParameters['Heavy_HH_18to39'].item()*profiles['HH_18to39'] + BioParameters['Heavy_HH_60plus'].item()*profiles['HH_60plus'] + BioParameters['Heavy_Parkingspot'].item()*profiles['Parkingspot']
U_HEV_plain = BioParameters['HEV_FracAdultHighedu'].item()*profiles['FracAdultHighedu'] + BioParameters['HEV_HH_inc_3high'].item()*profiles['HH_inc_high'] + BioParameters['HEV_HH_60plus'].item()*profiles['HH_60plus'] + BioParameters['HEV_Parkingspot'].item()*profiles['Parkingspot']

#The utility of each car type combination for the one-car class
U_onecar_stdlight = BioParameters['ASC_std_light'].item() + U_std + U_light
U_onecar_stdmidlight = BioParameters['ASC_Std_midlight'].item() + U_std + U_midlight
U_onecar_stdmidheavy = 0
U_onecar_stdheavy = BioParameters['ASC_Std_heavy'].item() + U_std + U_heavy
U_onecar_dieselmidlight = BioParameters['ASC_Diesel_midlight'].item() + U_diesel + U_midlight
U_onecar_dieselmidheavy = BioParameters['ASC_Diesel_midheavy'].item() + U_diesel + U_midheavy
U_onecar_dieselheavy = BioParameters['ASC_Diesel_heavy'].item() + U_diesel + U_heavy
U_onecar_HEV = BioParameters['ASC_HEV'].item() + U_HEV_plain

#The probability of each car type combination for the one-car class
profiles['P_onecar_stdlight'] = np.e**(U_onecar_stdlight)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))
profiles['P_onecar_stdmidlight'] = np.e**(U_onecar_stdmidlight)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))
profiles['P_onecar_stdmidheavy'] = np.e**(U_onecar_stdmidheavy)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))
profiles['P_onecar_stdheavy'] = np.e**(U_onecar_stdheavy)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))
profiles['P_onecar_dieselmidlight'] = np.e**(U_onecar_dieselmidlight)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))
profiles['P_onecar_dieselmidheavy'] = np.e**(U_onecar_dieselmidheavy)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))
profiles['P_onecar_dieselheavy'] = np.e**(U_onecar_dieselheavy)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))
profiles['P_onecar_HEV'] = np.e**(U_onecar_HEV)/(np.e**(U_onecar_stdlight) + np.e**(U_onecar_stdmidlight) + np.e**(U_onecar_stdmidheavy) + np.e**(U_onecar_stdheavy) + np.e**(U_onecar_dieselmidlight) + np.e**(U_onecar_dieselmidheavy) + np.e**(U_onecar_dieselheavy) + np.e**(U_onecar_HEV))

#The utility of each fuel type and weight category for the two-car class
U_twocar_std = U_std + BioParameters['Dummy_2car_std'].item()
U_twocar_diesel = U_diesel + BioParameters['Dummy_2car_diesel'].item()
U_twocar_light = U_light + BioParameters['Dummy_2car_light'].item()
U_twocar_midlight = U_midlight + BioParameters['Dummy_2car_midlight'].item()
U_twocar_midheavy = U_midheavy + BioParameters['Dummy_2car_midheavy'].item()
U_twocar_heavy = U_heavy + BioParameters['Dummy_2car_heavy'].item()
U_twocar_HEV = U_HEV_plain + BioParameters['Dummy_2car_HEV'].item()

#The utility of each car type combination for the two-car class
U_twocar_stdlight = BioParameters['ASC_std_light'].item() + U_twocar_std + U_twocar_light
U_twocar_stdmidlight = BioParameters['ASC_Std_midlight'].item() + U_twocar_std + U_twocar_midlight
U_twocar_stdmidheavy = 0
U_twocar_stdheavy = BioParameters['ASC_Std_heavy'].item() + U_twocar_std + U_twocar_heavy
U_twocar_dieselmidlight = BioParameters['ASC_Diesel_midlight'].item() + U_twocar_diesel + U_twocar_midlight
U_twocar_dieselmidheavy = BioParameters['ASC_Diesel_midheavy'].item() + U_twocar_diesel + U_twocar_midheavy
U_twocar_dieselheavy = BioParameters['ASC_Diesel_heavy'].item() + U_twocar_diesel + U_twocar_heavy
U_twocar_HEV = BioParameters['ASC_HEV'].item()+ U_twocar_HEV

#The probability of each car type combination for the two-car class
profiles['P_twocar_stdlight'] = np.e**(U_twocar_stdlight)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))
profiles['P_twocar_stdmidlight'] = np.e**(U_twocar_stdmidlight)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))
profiles['P_twocar_stdmidheavy'] = np.e**(U_twocar_stdmidheavy)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))
profiles['P_twocar_stdheavy'] = np.e**(U_twocar_stdheavy)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))
profiles['P_twocar_dieselmidlight'] = np.e**(U_twocar_dieselmidlight)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))
profiles['P_twocar_dieselmidheavy'] = np.e**(U_twocar_dieselmidheavy)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))
profiles['P_twocar_dieselheavy'] = np.e**(U_twocar_dieselheavy)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))
profiles['P_twocar_HEV'] = np.e**(U_twocar_HEV)/(np.e**(U_twocar_stdlight) + np.e**(U_twocar_stdmidlight) + np.e**(U_twocar_stdmidheavy) + np.e**(U_twocar_stdheavy) + np.e**(U_twocar_dieselmidlight) + np.e**(U_twocar_dieselmidheavy) + np.e**(U_twocar_dieselheavy) + np.e**(U_twocar_HEV))

profiles['P_stdlight'] = (profiles['Ponecar']*profiles['P_onecar_stdlight'] + profiles['Ptwocar']*profiles['P_twocar_stdlight'])/(profiles['Ponecar'] + profiles['Ptwocar'])
profiles['P_stdmidlight'] = (profiles['Ponecar']*profiles['P_onecar_stdmidlight'] + profiles['Ptwocar']*profiles['P_twocar_stdmidlight'])/(profiles['Ponecar'] + profiles['Ptwocar'])
profiles['P_stdmidheavy'] = (profiles['Ponecar']*profiles['P_onecar_stdmidheavy'] + profiles['Ptwocar']*profiles['P_twocar_stdmidheavy'])/(profiles['Ponecar'] + profiles['Ptwocar'])
profiles['P_stdheavy'] = (profiles['Ponecar']*profiles['P_onecar_stdheavy'] + profiles['Ptwocar']*profiles['P_twocar_stdheavy'])/(profiles['Ponecar'] + profiles['Ptwocar'])
profiles['P_dieselmidlight'] = (profiles['Ponecar']*profiles['P_onecar_dieselmidlight'] + profiles['Ptwocar']*profiles['P_twocar_dieselmidlight'])/(profiles['Ponecar'] + profiles['Ptwocar'])
profiles['P_dieselmidheavy'] = (profiles['Ponecar']*profiles['P_onecar_dieselmidheavy'] + profiles['Ptwocar']*profiles['P_twocar_dieselmidheavy'])/(profiles['Ponecar'] + profiles['Ptwocar'])
profiles['P_dieselheavy'] = (profiles['Ponecar']*profiles['P_onecar_dieselheavy'] + profiles['Ptwocar']*profiles['P_twocar_dieselheavy'])/(profiles['Ponecar'] + profiles['Ptwocar'])
profiles['P_HEV'] = (profiles['Ponecar']*profiles['P_onecar_HEV'] + profiles['Ptwocar']*profiles['P_twocar_HEV'])/(profiles['Ponecar'] + profiles['Ptwocar'])


'''XXXXX Computing the sampling-based energy consumption XXXXX'''

Vehicles = MPN[MPN['CarValid'] == 1] #To get the 3498 vehicles
Vehicles.loc[Vehicles['HH_cars'] < 2, 'Prob_carisusedmost'] = 1 #To be sure

Vehicles['SEC_times_weight'] = Vehicles['SEC_Travelcard_TNO']*Vehicles['Prob_carisusedmost'] #get sum(values*weights) for each household
SEC_times_weight_sum = Vehicles.groupby('HHID', as_index=False)['SEC_times_weight'].sum()
SEC_times_weight_sum.columns = ['HHID','SEC_times_weight_sum']
Vehicles = pd.merge(Vehicles, SEC_times_weight_sum, left_on='HHID', right_on='HHID', how = 'left') #Merge back into main dataset using HHID

Weight_sum = Vehicles.groupby('HHID', as_index=False)['Prob_carisusedmost'].sum() #get sum(weights) for each household
Weight_sum.columns = ['HHID','Weight_sum']
Vehicles = pd.merge(Vehicles, Weight_sum, left_on='HHID', right_on='HHID', how = 'left') #Merge back into main dataset using HHID

Vehicles['SEC_Travelcard_TNO_weighted'] = Vehicles['SEC_times_weight_sum']/Vehicles['Weight_sum'] #Note: if only one vehicle is valid, the weighted SEC now becomes the SEC of that vehicle 

##To get the mean and std SEC for the entire sample
# Vehicles_HHIDunique = Vehicles.drop_duplicates(subset = 'HHID') #Do not double-count the multicar households
# Mean_real = Vehicles_HHIDunique['SEC_Travelcard_TNO_weighted'].mean()
# Std_real = Vehicles_HHIDunique['SEC_Travelcard_TNO_weighted'].std()

Scale_factor = 10**9 #The code below creates integers, so a scale factor is included to transform the results into floats. 

Stdlight = Vehicles[Vehicles['Type'] == 11] 
Stdlight = Stdlight['SEC_Travelcard_TNO'].median()*Scale_factor #Draw the median to prevent overt influence from outliers (BEVs, SUVs, and oldtimers). Do NOT use the weighted mean: that mixes up the energy of the car types.
Stdmidlight = Vehicles[Vehicles['Type'] == 12] 
Stdmidlight = Stdmidlight['SEC_Travelcard_TNO'].median()*Scale_factor
Stdmidheavy = Vehicles[Vehicles['Type'] == 13]
Stdmidheavy = Stdmidheavy['SEC_Travelcard_TNO'].median()*Scale_factor
Stdheavy = Vehicles[Vehicles['Type'] == 14]
Stdheavy = Stdheavy['SEC_Travelcard_TNO'].median()*Scale_factor
Dieselmidlight = Vehicles[Vehicles['Type'] == 22]
Dieselmidlight = Dieselmidlight['SEC_Travelcard_TNO'].median()*Scale_factor
Dieselmidheavy = Vehicles[Vehicles['Type'] == 23]
Dieselmidheavy = Dieselmidheavy['SEC_Travelcard_TNO'].median()*Scale_factor
Dieselheavy = Vehicles[Vehicles['Type'] == 24]
Dieselheavy = Dieselheavy['SEC_Travelcard_TNO'].median()*Scale_factor
HEV = Vehicles[Vehicles['Type'] == 5]
HEV = HEV['SEC_Travelcard_TNO'].median()*Scale_factor

import random
SEClist = [Stdlight,Stdmidlight,Stdmidheavy,Stdheavy,Dieselmidlight,Dieselmidheavy,Dieselheavy,HEV]
profiles['SEC_draw'] = 0
for ind in profiles.index:
    probabilities_item = [profiles['P_stdlight'][ind],profiles['P_stdmidlight'][ind],profiles['P_stdmidheavy'][ind],profiles['P_stdheavy'][ind],profiles['P_dieselmidlight'][ind],profiles['P_dieselmidheavy'][ind],profiles['P_dieselheavy'][ind],profiles['P_HEV'][ind]]
    profiles['SEC_draw'][ind] = random.choices(population = SEClist, weights = probabilities_item)[0]
    
profiles['SEC_draw'] = profiles['SEC_draw']/Scale_factor #Use of Scale_factor to deconvert integers to floats
Prediction_mean = profiles['SEC_draw'].mean() #The prediction based on the DCM
Prediction_std = profiles['SEC_draw'].std() #The prediction based on the DCM
