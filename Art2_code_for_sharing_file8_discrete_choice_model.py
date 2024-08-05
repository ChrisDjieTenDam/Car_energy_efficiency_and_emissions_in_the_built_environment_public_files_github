# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 12:28:57 2022 by Chris ten Dam.
This code has been written for an academic study titled "Car energy efficiency and emissions in the built environment".

The first part contains some final data cleaning steps. 
Next, the discrete choice modeling framework is estimated. 
"""

import os
data_directory = r"Q:\research-driving-energy"

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, linear_model
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.loglikelihood as ll
from biogeme import models
import biogeme.distributions as dist
from biogeme.expressions import Beta, Elem, exp, log#, MonteCarlo, bioDraws

MPN = pd.read_csv(data_directory + os.sep + 'MPN_processed_data/MPN3years_1entrypercar.csv', index_col = 0) #The main file 
MPN.loc[MPN['HH_cars'] > 1,'HH_weight'] = MPN['HH_weight']*MPN['Prob_carisusedmost'] #Prob = 1 for households with one (or no) car
MPN = MPN.astype(float) #To stop the models from crashing

MPN['HH_weight'] = MPN['HH_weight']*1.1316046025500308

train = MPN[['HHID','HH_cars','HH_weight','Type','CarValid','HH_under12','HH_12to17','HH_18to39','HH_40to59','HH_60plus','FracAdultMales','FracAdultWorkers','FracAdultHighedu','HH_inc_real','Density_PC5','km_center','km_mediumcenter','km_largercenter','km_hugecenter','Landuse','NDVI','Parkingspot','km_super','km_station','km_bigstation','km_bus']]
train = train.dropna()

train['HH_under18'] = train['HH_under12'] + train['HH_12to17']
train['HH_18to59'] = train['HH_18to39'] + train['HH_40to59']

#The mean real income of each class
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
train_HHIDunique = train.drop_duplicates(subset = 'HHID') #4316 households to match descriptive statistics

standard_scaler = preprocessing.StandardScaler() 
standard_scaler.fit(train_HHIDunique[xvars])
train[xvars] = standard_scaler.transform(train[xvars])

train = train[['HH_cars','HH_weight','Type','CarValid','HH_under18','HH_18to59','HH_18to39','HH_40to59','HH_60plus','FracAdultMales','FracAdultWorkers','FracAdultHighedu','HH_inc_low', 'HH_inc_middle','HH_inc_middlehigh', 'HH_inc_high','Density_PC5','km_center','km_mediumcenter','km_largercenter','km_hugecenter','Landuse','NDVI','Parkingspot','km_super','km_station','km_bigstation','km_bus']]
database = db.Database('train', train)
globals().update(database.variables)

'''XXXXX Computing the Variance Inflation Factors XXXXX '''

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

train_vif = train[['FracAdultMales','FracAdultWorkers','FracAdultHighedu','HH_under18','HH_18to39','HH_40to59','HH_60plus','HH_inc_low','HH_inc_middle','HH_inc_middlehigh','HH_inc_high','Density_PC5','km_largercenter','km_hugecenter','NDVI','Parkingspot','Landuse','km_bus','km_station','km_bigstation']] #Max VIF = 3.1 for NDVI Max VIF = 1.3 for NDVI when only keeping undertwelve, highedu, adults, income, NDVI, parkign, and logcenter
vif2 = pd.DataFrame() 
vif2["VIF Factor"] = [variance_inflation_factor(train_vif.values, i) for i in range(train_vif.shape[1])]
vif2["features"] = train_vif.columns #14 Vars: 3.2 for NDVI and 2.7 for density. 21 Vars: 5.5 for HH_people, 5.2 for allEur, 3.9 for HH_undertwelve, 3.8 for density, 3.3 for NDVI. Risky but maybe acceptable


'''XXXXX Defining the parameters XXXXX''' 

#Classification utility
OwnscarConstant = Beta('OwnscarConstant', 0, None, None, 0)
Ownscar_FracAdultMales = Beta('Ownscar_FracAdultMales', 0, None, None, 1) 
Ownscar_FracAdultHighedu = Beta('Ownscar_FracAdultHighedu', 0, None, None, 1) 
Ownscar_FracAdultWorkers = Beta('Ownscar_FracAdultWorkers', 0, None, None, 0)
Ownscar_HH_00under18 = Beta('Ownscar_HH_00under18', 0, None, None, 0)
Ownscar_HH_18to39 = Beta('Ownscar_HH_18to39', 0, None, None, 0)
Ownscar_HH_40to59 = Beta('Ownscar_HH_40to59', 0, None, None, 0)
Ownscar_HH_18to59 = Beta('Ownscar_HH_18to59', 0, None, None, 1)
Ownscar_HH_60plus = Beta('Ownscar_HH_60plus', 0, None, None, 0)
Ownscar_HH_inc_0low = Beta('Ownscar_HH_inc_0low', 0, None, None, 0)
Ownscar_HH_inc_1middle = Beta('Ownscar_HH_inc_1middle', 0, None, None, 0)
Ownscar_HH_inc_2middlehigh = Beta('Ownscar_HH_inc_2middlehigh', 0, None, None, 0)
Ownscar_HH_inc_3high = Beta('Ownscar_HH_inc_3high', 0, None, None, 0) 
Ownscar_Density_PC5 = Beta('Ownscar_Density_PC5', 0, None, None, 0)
Ownscar_km_center = Beta('Ownscar_km_center', 0, None, None, 1)
Ownscar_km_mediumcenter = Beta('Ownscar_km_mediumcenter', 0, None, None, 1) 
Ownscar_km_largercenter = Beta('Ownscar_km_largercenter', 0, None, None, 1) 
Ownscar_km_hugecenter = Beta('Ownscar_km_hugecenter', 0, None, None, 0)
Ownscar_Landuse = Beta('Ownscar_Landuse', 0, None, None, 1) 
Ownscar_NDVI = Beta('Ownscar_NDVI', 0, None, None, 1) 
Ownscar_Parkingspot = Beta('Ownscar_Parkingspot', 0, None, None, 0) 
Ownscar_km_super = Beta('Ownscar_km_super', 0, None, None, 1)
Ownscar_km_station = Beta('Ownscar_km_station', 0, None, None, 0) 
Ownscar_km_bigstation = Beta('Ownscar_km_bigstation', 0, None, None, 0)
Ownscar_km_bus = Beta('Ownscar_km_bus', 0, None, None, 0)

TwocarConstant = Beta('TwocarConstant', 0, None, None, 0)
Twocar_FracAdultMales = Beta('Twocar_FracAdultMales', 0, None, None, 0)
Twocar_FracAdultHighedu = Beta('Twocar_FracAdultHighedu', 0, None, None, 1) 
Twocar_FracAdultWorkers = Beta('Twocar_FracAdultWorkers', 0, None, None, 0)
Twocar_HH_00under18 = Beta('Twocar_HH_00under18', 0, None, None, 0)
Twocar_HH_18to39 = Beta('Twocar_HH_18to39', 0, None, None, 0)
Twocar_HH_40to59 = Beta('Twocar_HH_40to59', 0, None, None, 0)
Twocar_HH_18to59 = Beta('Twocar_HH_18to59', 0, None, None, 1)
Twocar_HH_60plus = Beta('Twocar_HH_60plus', 0, None, None, 0)
Twocar_HH_inc_0low = Beta('Twocar_HH_inc_0low', 0, None, None, 0)
Twocar_HH_inc_1middle = Beta('Twocar_HH_inc_1middle', 0, None, None, 0)
Twocar_HH_inc_2middlehigh = Beta('Twocar_HH_inc_2middlehigh', 0, None, None, 0)
Twocar_HH_inc_3high = Beta('Twocar_HH_inc_3high', 0, None, None, 0)
Twocar_Density_PC5 = Beta('Twocar_Density_PC5', 0, None, None, 0)
Twocar_km_center = Beta('Twocar_km_center', 0, None, None, 1)
Twocar_km_mediumcenter = Beta('Twocar_km_mediumcenter', 0, None, None, 1) 
Twocar_km_largercenter = Beta('Twocar_km_largercenter', 0, None, None, 1) 
Twocar_km_hugecenter = Beta('Twocar_km_hugecenter', 0, None, None, 0) 
Twocar_Landuse = Beta('Twocar_Landuse', 0, None, None, 1) 
Twocar_NDVI = Beta('Twocar_NDVI', 0, None, None, 1) 
Twocar_Parkingspot = Beta('Twocar_Parkingspot', 0, None, None, 0) 
Twocar_km_super = Beta('Twocar_km_super', 0, None, None, 1)
Twocar_km_station = Beta('Twocar_km_station', 0, None, None, 0)
Twocar_km_bigstation = Beta('Twocar_km_bigstation', 0, None, None, 0)
Twocar_km_bus = Beta('Twocar_km_bus', 0, None, None, 0)

#Weight- and fuel-based additions to cartype utility
Std_FracAdultMales = Beta('Std_FracAdultMales', 0, None, None, 0)
Std_FracAdultHighedu = Beta('Std_FracAdultHighedu', 0, None, None, 0) 
Std_FracAdultWorkers = Beta('Std_FracAdultWorkers', 0, None, None, 1) 
Std_HH_00under18 = Beta('Std_HH_00under18', 0, None, None, 1) 
Std_HH_18to39 = Beta('Std_HH_18to39', 0, None, None, 0) 
Std_HH_40to59 = Beta('Std_HH_40to59', 0, None, None, 1) 
Std_HH_18to59 = Beta('Std_HH_18to59', 0, None, None, 1)
Std_HH_60plus = Beta('Std_HH_60plus', 0, None, None, 1) 
Std_HH_inc_0low = Beta('Std_HH_inc_0low', 0, None, None, 0)
Std_HH_inc_1middle = Beta('Std_HH_inc_1middle', 0, None, None, 1) 
Std_HH_inc_2middlehigh = Beta('Std_HH_inc_2middlehigh', 0, None, None, 1) 
Std_HH_inc_3high = Beta('Std_HH_inc_3high', 0, None, None, 1) 
Std_Density_PC5 = Beta('Std_Density_PC5', 0, None, None, 1) 
Std_km_center = Beta('Std_km_center', 0, None, None, 1)
Std_km_mediumcenter = Beta('Std_km_mediumcenter', 0, None, None, 1)
Std_km_largercenter = Beta('Std_km_largercenter', 0, None, None, 0)
Std_km_hugecenter = Beta('Std_km_hugecenter', 0, None, None, 0)
Std_Landuse = Beta('Std_Landuse', 0, None, None, 1) 
Std_NDVI = Beta('Std_NDVI', 0, None, None, 1) 
Std_Parkingspot = Beta('Std_Parkingspot', 0, None, None, 1) 
Std_km_super = Beta('Std_km_super', 0, None, None, 1)
Std_km_station = Beta('Std_km_station', 0, None, None, 1) 
Std_km_bigstation = Beta('Std_km_bigstation', 0, None, None, 1)
Std_km_bus = Beta('Std_km_bus', 0, None, None, 1) 

Diesel_FracAdultMales = Beta('Diesel_FracAdultMales', 0, None, None, 1) 
Diesel_FracAdultHighedu = Beta('Diesel_FracAdultHighedu', 0, None, None, 1) 
Diesel_FracAdultWorkers = Beta('Diesel_FracAdultWorkers', 0, None, None, 0)
Diesel_HH_00under18 = Beta('Diesel_HH_00under18', 0, None, None, 1) 
Diesel_HH_18to39 = Beta('Diesel_HH_18to39', 0, None, None, 0)
Diesel_HH_40to59 = Beta('Diesel_HH_40to59', 0, None, None, 0)
Diesel_HH_18to59 = Beta('Diesel_HH_18to59', 0, None, None, 1)
Diesel_HH_60plus = Beta('Diesel_HH_60plus', 0, None, None, 1) 
Diesel_HH_inc_0low = Beta('Diesel_HH_inc_0low', 0, None, None, 1) 
Diesel_HH_inc_1middle = Beta('Diesel_HH_inc_1middle', 0, None, None, 1) 
Diesel_HH_inc_2middlehigh = Beta('Diesel_HH_inc_2middlehigh', 0, None, None, 1) 
Diesel_HH_inc_3high = Beta('Diesel_HH_inc_3high', 0, None, None, 1) 
Diesel_Density_PC5 = Beta('Diesel_Density_PC5', 0, None, None, 1) 
Diesel_km_center = Beta('Diesel_km_center', 0, None, None, 1)
Diesel_km_mediumcenter = Beta('Diesel_km_mediumcenter', 0, None, None, 1)
Diesel_km_largercenter = Beta('Diesel_km_largercenter', 0, None, None, 0)
Diesel_km_hugecenter = Beta('Diesel_km_hugecenter', 0, None, None, 1) 
Diesel_Landuse = Beta('Diesel_Landuse', 0, None, None, 0)
Diesel_NDVI = Beta('Diesel_NDVI', 0, None, None, 1) 
Diesel_Parkingspot = Beta('Diesel_Parkingspot', 0, None, None, 0)
Diesel_km_super = Beta('Diesel_km_super', 0, None, None, 1)
Diesel_km_station = Beta('Diesel_km_station', 0, None, None, 1) 
Diesel_km_bigstation = Beta('Diesel_km_bigstation', 0, None, None, 1)
Diesel_km_bus = Beta('Diesel_km_bus', 0, None, None, 1) 

HEV_FracAdultMales = Beta('HEV_FracAdultMales', 0, None, None, 1) 
HEV_FracAdultHighedu = Beta('HEV_FracAdultHighedu', 0, None, None, 0)
HEV_FracAdultWorkers = Beta('HEV_FracAdultWorkers', 0, None, None, 1) 
HEV_HH_00under18 = Beta('HEV_HH_00under18', 0, None, None, 1) 
HEV_HH_18to39 = Beta('HEV_HH_18to39', 0, None, None, 1)  
HEV_HH_40to59 = Beta('HEV_HH_40to59', 0, None, None, 1)  
HEV_HH_18to59 = Beta('HEV_HH_18to59', 0, None, None, 1)
HEV_HH_60plus = Beta('HEV_HH_60plus', 0, None, None, 0) 
HEV_HH_inc_0low = Beta('HEV_HH_inc_0low', 0, None, None, 1) 
HEV_HH_inc_1middle = Beta('HEV_HH_inc_1middle', 0, None, None, 1) 
HEV_HH_inc_2middlehigh = Beta('HEV_HH_inc_2middlehigh', 0, None, None, 1) 
HEV_HH_inc_3high = Beta('HEV_HH_inc_3high', 0, None, None, 0)
HEV_Density_PC5 = Beta('HEV_Density_PC5', 0, None, None, 1) 
HEV_km_center = Beta('HEV_km_center', 0, None, None, 1)
HEV_km_mediumcenter = Beta('HEV_km_mediumcenter', 0, None, None, 1)
HEV_km_largercenter = Beta('HEV_km_largercenter', 0, None, None, 1) 
HEV_km_hugecenter = Beta('HEV_km_hugecenter', 0, None, None, 1) 
HEV_Landuse = Beta('HEV_Landuse', 0, None, None, 1) 
HEV_NDVI = Beta('HEV_NDVI', 0, None, None, 1) 
HEV_Parkingspot = Beta('HEV_Parkingspot', 0, None, None, 0) 
HEV_km_super = Beta('HEV_km_super', 0, None, None, 1)
HEV_km_station = Beta('HEV_km_station', 0, None, None, 1) 
HEV_km_bigstation = Beta('HEV_km_bigstation', 0, None, None, 1)
HEV_km_bus = Beta('HEV_km_bus', 0, None, None, 1) 

Light_FracAdultMales = Beta('Light_FracAdultMales', 0, None, None, 0)
Light_FracAdultHighedu = Beta('Light_FracAdultHighedu', 0, None, None, 1) 
Light_FracAdultWorkers = Beta('Light_FracAdultWorkers', 0, None, None, 1) 
Light_HH_00under18 = Beta('Light_HH_00under18', 0, None, None, 0)
Light_HH_18to39 = Beta('Light_HH_18to39', 0, None, None, 1) 
Light_HH_40to59 = Beta('Light_HH_40to59', 0, None, None, 0)
Light_HH_18to59 = Beta('Light_HH_18to59', 0, None, None, 1)
Light_HH_60plus = Beta('Light_HH_60plus', 0, None, None, 0)
Light_HH_inc_0low = Beta('Light_HH_inc_0low', 0, None, None, 1) 
Light_HH_inc_1middle = Beta('Light_HH_inc_1middle', 0, None, None, 0)
Light_HH_inc_2middlehigh = Beta('Light_HH_inc_2middlehigh', 0, None, None, 0)
Light_HH_inc_3high = Beta('Light_HH_inc_3high', 0, None, None, 0) 
Light_Density_PC5 = Beta('Light_Density_PC5', 0, None, None, 1) 
Light_km_center = Beta('Light_km_center', 0, None, None, 1)
Light_km_mediumcenter = Beta('Light_km_mediumcenter', 0, None, None, 1)
Light_km_largercenter = Beta('Light_km_largercenter', 0, None, None, 1) 
Light_km_hugecenter = Beta('Light_km_hugecenter', 0, None, None, 0)
Light_Landuse = Beta('Light_Landuse', 0, None, None, 1) 
Light_NDVI = Beta('Light_NDVI', 0, None, None, 0) 
Light_Parkingspot = Beta('Light_Parkingspot', 0, None, None, 0) 
Light_km_super = Beta('Light_km_super', 0, None, None, 1)
Light_km_station = Beta('Light_km_station', 0, None, None, 1) 
Light_km_bigstation = Beta('Light_km_bigstation', 0, None, None, 1)
Light_km_bus = Beta('Light_km_bus', 0, None, None, 1) 

Midlight_FracAdultMales = Beta('Midlight_FracAdultMales', 0, None, None, 1) 
Midlight_FracAdultHighedu = Beta('Midlight_FracAdultHighedu', 0, None, None, 1) 
Midlight_FracAdultWorkers = Beta('Midlight_FracAdultWorkers', 0, None, None, 0)
Midlight_HH_00under18 = Beta('Midlight_HH_00under18', 0, None, None, 0)
Midlight_HH_18to39 = Beta('Midlight_HH_18to39', 0, None, None, 1) 
Midlight_HH_40to59 = Beta('Midlight_HH_40to59', 0, None, None, 1) 
Midlight_HH_18to59 = Beta('Midlight_HH_18to59', 0, None, None, 1)
Midlight_HH_60plus = Beta('Midlight_HH_60plus', 0, None, None, 0) 
Midlight_HH_inc_0low = Beta('Midlight_HH_inc_0low', 0, None, None, 1) 
Midlight_HH_inc_1middle = Beta('Midlight_HH_inc_1middle', 0, None, None, 1) 
Midlight_HH_inc_2middlehigh = Beta('Midlight_HH_inc_2middlehigh', 0, None, None, 1) 
Midlight_HH_inc_3high = Beta('Midlight_HH_inc_3high', 0, None, None, 0) 
Midlight_Density_PC5 = Beta('Midlight_Density_PC5', 0, None, None, 1) 
Midlight_km_center = Beta('Midlight_km_center', 0, None, None, 1)
Midlight_km_mediumcenter = Beta('Midlight_km_mediumcenter', 0, None, None, 1)
Midlight_km_largercenter = Beta('Midlight_km_largercenter', 0, None, None, 1) 
Midlight_km_hugecenter = Beta('Midlight_km_hugecenter', 0, None, None, 0)
Midlight_Landuse = Beta('Midlight_Landuse', 0, None, None, 1) 
Midlight_NDVI = Beta('Midlight_NDVI', 0, None, None, 1) 
Midlight_Parkingspot = Beta('Midlight_Parkingspot', 0, None, None, 1) 
Midlight_km_super = Beta('Midlight_km_super', 0, None, None, 1)
Midlight_km_station = Beta('Midlight_km_station', 0, None, None, 1) 
Midlight_km_bigstation = Beta('Midlight_km_bigstation', 0, None, None, 1)
Midlight_km_bus = Beta('Midlight_km_bus', 0, None, None, 1) 

Midheavy_FracAdultMales = Beta('Midheavy_FracAdultMales', 0, None, None, 1) 
Midheavy_FracAdultHighedu = Beta('Midheavy_FracAdultHighedu', 0, None, None, 1) 
Midheavy_FracAdultWorkers = Beta('Midheavy_FracAdultWorkers', 0, None, None, 0)
Midheavy_HH_00under18 = Beta('Midheavy_HH_00under18', 0, None, None, 1)   
Midheavy_HH_18to39 = Beta('Midheavy_HH_18to39', 0, None, None, 1) 
Midheavy_HH_40to59 = Beta('Midheavy_HH_40to59', 0, None, None, 0) 
Midheavy_HH_18to59 = Beta('Midheavy_HH_18to59', 0, None, None, 1)
Midheavy_HH_60plus = Beta('Midheavy_HH_60plus', 0, None, None, 1) 
Midheavy_HH_inc_0low = Beta('Midheavy_HH_inc_0low', 0, None, None, 1) 
Midheavy_HH_inc_1middle = Beta('Midheavy_HH_inc_1middle', 0, None, None, 1) 
Midheavy_HH_inc_2middlehigh = Beta('Midheavy_HH_inc_2middlehigh', 0, None, None, 0)
Midheavy_HH_inc_3high = Beta('Midheavy_HH_inc_3high', 0, None, None, 1) 
Midheavy_Density_PC5 = Beta('Midheavy_Density_PC5', 0, None, None, 1) 
Midheavy_km_center = Beta('Midheavy_km_center', 0, None, None, 1)
Midheavy_km_mediumcenter = Beta('Midheavy_km_mediumcenter', 0, None, None, 1)
Midheavy_km_largercenter = Beta('Midheavy_km_largercenter', 0, None, None, 1)  
Midheavy_km_hugecenter = Beta('Midheavy_km_hugecenter', 0, None, None, 1) 
Midheavy_Landuse = Beta('Midheavy_Landuse', 0, None, None, 1) 
Midheavy_NDVI = Beta('Midheavy_NDVI', 0, None, None, 0) 
Midheavy_Parkingspot = Beta('Midheavy_Parkingspot', 0, None, None, 1) 
Midheavy_km_super = Beta('Midheavy_km_super', 0, None, None, 1)
Midheavy_km_station = Beta('Midheavy_km_station', 0, None, None, 1) 
Midheavy_km_bigstation = Beta('Midheavy_km_bigstation', 0, None, None, 1)
Midheavy_km_bus = Beta('Midheavy_km_bus', 0, None, None, 1) 

Heavy_FracAdultMales = Beta('Heavy_FracAdultMales', 0, None, None, 0)
Heavy_FracAdultHighedu = Beta('Heavy_FracAdultHighedu', 0, None, None, 0) 
Heavy_FracAdultWorkers = Beta('Heavy_FracAdultWorkers', 0, None, None, 1)  
Heavy_HH_00under18 = Beta('Heavy_HH_00under18', 0, None, None, 0) 
Heavy_HH_18to39 = Beta('Heavy_HH_18to39', 0, None, None, 0) 
Heavy_HH_40to59 = Beta('Heavy_HH_40to59', 0, None, None, 1) 
Heavy_HH_18to59 = Beta('Heavy_HH_18to59', 0, None, None, 1)
Heavy_HH_60plus = Beta('Heavy_HH_60plus', 0, None, None, 0) 
Heavy_HH_inc_0low = Beta('Heavy_HH_inc_0low', 0, None, None, 1) 
Heavy_HH_inc_1middle = Beta('Heavy_HH_inc_1middle', 0, None, None, 1) 
Heavy_HH_inc_2middlehigh = Beta('Heavy_HH_inc_2middlehigh', 0, None, None, 1) 
Heavy_HH_inc_3high = Beta('Heavy_HH_inc_3high', 0, None, None, 1) 
Heavy_Density_PC5 = Beta('Heavy_Density_PC5', 0, None, None, 1) 
Heavy_km_center = Beta('Heavy_km_center', 0, None, None, 1)
Heavy_km_mediumcenter = Beta('Heavy_km_mediumcenter', 0, None, None, 1)
Heavy_km_largercenter = Beta('Heavy_km_largercenter', 0, None, None, 1) 
Heavy_km_hugecenter = Beta('Heavy_km_hugecenter', 0, None, None, 1) 
Heavy_Landuse = Beta('Heavy_Landuse', 0, None, None, 1)
Heavy_NDVI = Beta('Heavy_NDVI', 0, None, None, 1) 
Heavy_Parkingspot = Beta('Heavy_Parkingspot', 0, None, None, 0) 
Heavy_km_super = Beta('Heavy_km_super', 0, None, None, 1)
Heavy_km_station = Beta('Heavy_km_station', 0, None, None, 1) 
Heavy_km_bigstation = Beta('Heavy_km_bigstation', 0, None, None, 1)
Heavy_km_bus = Beta('Heavy_km_bus', 0, None, None, 1) 

ASC_std_light = Beta('ASC_std_light', 0, None, None, 0)
ASC_Std_midlight = Beta('ASC_Std_midlight', 0, None, None, 0)
ASC_Std_midheavy = Beta('ASC_Std_midheavy', 0, None, None, 0)
ASC_Std_heavy = Beta('ASC_Std_heavy', 0, None, None, 0)
ASC_Diesel_midlight = Beta('ASC_Diesel_midlight', 0, None, None, 0)
ASC_Diesel_midheavy = Beta('ASC_Diesel_midheavy', 0, None, None, 0)
ASC_Diesel_heavy = Beta('ASC_Diesel_heavy', 0, None, None, 0)
ASC_HEV = Beta('ASC_HEV', 0, None, None, 0)

Dummy_2car_std = Beta('Dummy_2car_std', 0, None, None, 0) 
Dummy_2car_diesel = Beta('Dummy_2car_diesel', 0, None, None, 0)
Dummy_2car_light = Beta('Dummy_2car_light', 0, None, None, 0)
Dummy_2car_midlight = Beta('Dummy_2car_midlight', 0, None, None, 0) 
Dummy_2car_midheavy = Beta('Dummy_2car_midheavy', 0, None, None, 0) 
Dummy_2car_heavy = Beta('Dummy_2car_heavy', 0, None, None, 0)
Dummy_2car_HEV = Beta('Dummy_2car_HEV', 0, None, None, 0)


'''XXXXX The Structural Equation Model XXXXX'''

#Multinomial model of car ownership as the determinants of whether or not people own a car may differ from the determinants of whether people own two cars
Utility_Carless = 0
Utility_Ownscar = OwnscarConstant + Ownscar_FracAdultMales*FracAdultMales + Ownscar_FracAdultHighedu*FracAdultHighedu + Ownscar_FracAdultWorkers*FracAdultWorkers + Ownscar_HH_00under18*HH_under18 + Ownscar_HH_18to39*HH_18to39 + Ownscar_HH_40to59*HH_40to59 + Ownscar_HH_18to59*HH_18to59 + Ownscar_HH_60plus*HH_60plus + Ownscar_HH_inc_0low*HH_inc_low + Ownscar_HH_inc_1middle*HH_inc_middle  + Ownscar_HH_inc_2middlehigh*HH_inc_middlehigh + Ownscar_HH_inc_3high*HH_inc_high + Ownscar_Density_PC5*Density_PC5 + Ownscar_km_center*km_center + Ownscar_km_mediumcenter*km_mediumcenter + Ownscar_km_largercenter*km_largercenter + Ownscar_km_hugecenter*km_hugecenter + Ownscar_Landuse*Landuse + Ownscar_NDVI*NDVI + Ownscar_km_super*km_super + Ownscar_km_station*km_station + Ownscar_km_bigstation*km_bigstation + Ownscar_km_bus*km_bus + Ownscar_Parkingspot*Parkingspot
Utility_Twocar = TwocarConstant + Twocar_FracAdultMales*FracAdultMales + Twocar_FracAdultHighedu*FracAdultHighedu + Twocar_FracAdultWorkers*FracAdultWorkers + Twocar_HH_00under18*HH_under18 + Twocar_HH_18to39*HH_18to39 + Twocar_HH_40to59*HH_40to59 + Twocar_HH_18to59*HH_18to59 + Twocar_HH_60plus*HH_60plus + Twocar_HH_inc_0low*HH_inc_low + Twocar_HH_inc_1middle*HH_inc_middle + Twocar_HH_inc_2middlehigh*HH_inc_middlehigh + Twocar_HH_inc_3high*HH_inc_high + Twocar_Density_PC5*Density_PC5 + Twocar_km_center*km_center +  Twocar_km_mediumcenter*km_mediumcenter + Twocar_km_largercenter*km_largercenter + Twocar_km_hugecenter*km_hugecenter + Twocar_Landuse*Landuse + Twocar_NDVI*NDVI + Twocar_km_super*km_super +  Twocar_km_station*km_station + Twocar_km_bigstation*km_bigstation + Twocar_km_bus*km_bus + Twocar_Parkingspot*Parkingspot

Vcarownership = {0: Utility_Carless, 1: Utility_Ownscar, 2: Utility_Twocar}
CarOwnProb =  models.logit(Vcarownership, None, HH_cars)

#The probability of being in each car ownership class as computed with the logit equation
Pcarless = exp(Utility_Carless)/(exp(Utility_Carless) + exp(Utility_Ownscar) + exp(Utility_Twocar))
Ponecar = exp(Utility_Ownscar)/(exp(Utility_Carless) + exp(Utility_Ownscar) + exp(Utility_Twocar))
Pmorecars = exp(Utility_Twocar)/(exp(Utility_Carless) + exp(Utility_Ownscar) + exp(Utility_Twocar))

#The utility associated with each fueltype and weight
U_std = Std_FracAdultMales*FracAdultMales + Std_FracAdultHighedu*FracAdultHighedu + Std_FracAdultWorkers*FracAdultWorkers + Std_HH_00under18*HH_under18 + Std_HH_18to39*HH_18to39 + Std_HH_40to59*HH_40to59 + Std_HH_18to59*HH_18to59 + Std_HH_60plus*HH_60plus + Std_HH_inc_0low*HH_inc_low + Std_HH_inc_1middle*HH_inc_middle + Std_HH_inc_2middlehigh*HH_inc_middlehigh + Std_HH_inc_3high*HH_inc_high + Std_Density_PC5*Density_PC5 + Std_km_center*km_center + Std_km_mediumcenter*km_mediumcenter + Std_km_largercenter*km_largercenter + Std_km_hugecenter*km_hugecenter + Std_Landuse*Landuse + Std_NDVI*NDVI + Std_km_super*km_super + Std_km_station*km_station + Std_km_bigstation*km_bigstation + Std_km_bus*km_bus + Std_Parkingspot*Parkingspot
U_diesel = Diesel_FracAdultMales*FracAdultMales + Diesel_FracAdultHighedu*FracAdultHighedu + Diesel_FracAdultWorkers*FracAdultWorkers + Diesel_HH_00under18*HH_under18 + Diesel_HH_18to39*HH_18to39 + Diesel_HH_40to59*HH_40to59 + Diesel_HH_18to59*HH_18to59 + Diesel_HH_60plus*HH_60plus + Diesel_HH_inc_0low*HH_inc_low + Diesel_HH_inc_1middle*HH_inc_middle + Diesel_HH_inc_2middlehigh*HH_inc_middlehigh + Diesel_HH_inc_3high*HH_inc_high + Diesel_Density_PC5*Density_PC5 + Diesel_km_center*km_center + Diesel_km_mediumcenter*km_mediumcenter + Diesel_km_largercenter*km_largercenter + Diesel_km_hugecenter*km_hugecenter + Diesel_Landuse*Landuse + Diesel_NDVI*NDVI + Diesel_km_super*km_super + Diesel_km_station*km_station + Diesel_km_bigstation*km_bigstation + Diesel_km_bus*km_bus + Diesel_Parkingspot*Parkingspot
U_light = Light_FracAdultMales*FracAdultMales + Light_FracAdultHighedu*FracAdultHighedu + Light_FracAdultWorkers*FracAdultWorkers + Light_HH_00under18*HH_under18 + Light_HH_18to39*HH_18to39 + Light_HH_40to59*HH_40to59 + Light_HH_18to59*HH_18to59 + Light_HH_60plus*HH_60plus + Light_HH_inc_0low*HH_inc_low + Light_HH_inc_1middle*HH_inc_middle + Light_HH_inc_2middlehigh*HH_inc_middlehigh + Light_HH_inc_3high*HH_inc_high + Light_Density_PC5*Density_PC5 + Light_km_center*km_center + Light_km_mediumcenter*km_mediumcenter + Light_km_largercenter*km_largercenter + Light_km_hugecenter*km_hugecenter + Light_Landuse*Landuse + Light_NDVI*NDVI + Light_km_super*km_super + Light_km_station*km_station + Light_km_bigstation*km_bigstation + Light_km_bus*km_bus + Light_Parkingspot*Parkingspot
U_midlight = Midlight_FracAdultMales*FracAdultMales + Midlight_FracAdultHighedu*FracAdultHighedu + Midlight_FracAdultWorkers*FracAdultWorkers + Midlight_HH_00under18*HH_under18 + Midlight_HH_18to39*HH_18to39 + Midlight_HH_40to59*HH_40to59 + Midlight_HH_18to59*HH_18to59 + Midlight_HH_60plus*HH_60plus + Midlight_HH_inc_0low*HH_inc_low + Midlight_HH_inc_1middle*HH_inc_middle + Midlight_HH_inc_2middlehigh*HH_inc_middlehigh + Midlight_HH_inc_3high*HH_inc_high + Midlight_Density_PC5*Density_PC5 + Midlight_km_center*km_center + Midlight_km_mediumcenter*km_mediumcenter + Midlight_km_largercenter*km_largercenter + Midlight_km_hugecenter*km_hugecenter + Midlight_Landuse*Landuse + Midlight_NDVI*NDVI + Midlight_km_super*km_super + Midlight_km_station*km_station + Midlight_km_bigstation*km_bigstation + Midlight_km_bus*km_bus + Midlight_Parkingspot*Parkingspot
U_midheavy = Midheavy_FracAdultMales*FracAdultMales + Midheavy_FracAdultHighedu*FracAdultHighedu + Midheavy_FracAdultWorkers*FracAdultWorkers + Midheavy_HH_00under18*HH_under18 + Midheavy_HH_18to39*HH_18to39 + Midheavy_HH_40to59*HH_40to59 + Midheavy_HH_18to59*HH_18to59 + Midheavy_HH_60plus*HH_60plus + Midheavy_HH_inc_0low*HH_inc_low + Midheavy_HH_inc_1middle*HH_inc_middle + Midheavy_HH_inc_2middlehigh*HH_inc_middlehigh + Midheavy_HH_inc_3high*HH_inc_high + Midheavy_Density_PC5*Density_PC5 + Midheavy_km_center*km_center + Midheavy_km_mediumcenter*km_mediumcenter + Midheavy_km_largercenter*km_largercenter + Midheavy_km_hugecenter*km_hugecenter + Midheavy_Landuse*Landuse + Midheavy_NDVI*NDVI + Midheavy_km_super*km_super + Midheavy_km_station*km_station + Midheavy_km_bigstation*km_bigstation + Midheavy_km_bus*km_bus + Midheavy_Parkingspot*Parkingspot
U_heavy = Heavy_FracAdultMales*FracAdultMales + Heavy_FracAdultHighedu*FracAdultHighedu + Heavy_FracAdultWorkers*FracAdultWorkers + Heavy_HH_00under18*HH_under18 + Heavy_HH_18to39*HH_18to39 + Heavy_HH_40to59*HH_40to59 + Heavy_HH_18to59*HH_18to59 + Heavy_HH_60plus*HH_60plus + Heavy_HH_inc_0low*HH_inc_low + Heavy_HH_inc_1middle*HH_inc_middle + Heavy_HH_inc_2middlehigh*HH_inc_middlehigh + Heavy_HH_inc_3high*HH_inc_high + Heavy_Density_PC5*Density_PC5 + Heavy_km_center*km_center + Heavy_km_mediumcenter*km_mediumcenter + Heavy_km_largercenter*km_largercenter + Heavy_km_hugecenter*km_hugecenter + Heavy_Landuse*Landuse + Heavy_NDVI*NDVI + Heavy_km_super*km_super + Heavy_km_station*km_station + Heavy_km_bigstation*km_bigstation + Heavy_km_bus*km_bus + Heavy_Parkingspot*Parkingspot
U_HEV_plain = HEV_FracAdultMales*FracAdultMales + HEV_FracAdultHighedu*FracAdultHighedu + HEV_FracAdultWorkers*FracAdultWorkers + HEV_HH_00under18*HH_under18 + HEV_HH_18to39*HH_18to39 + HEV_HH_40to59*HH_40to59 + HEV_HH_18to59*HH_18to59 + HEV_HH_60plus*HH_60plus + HEV_HH_inc_0low*HH_inc_low + HEV_HH_inc_1middle*HH_inc_middle + HEV_HH_inc_2middlehigh*HH_inc_middlehigh + HEV_HH_inc_3high*HH_inc_high + HEV_Density_PC5*Density_PC5 + HEV_km_center*km_center + HEV_km_mediumcenter*km_mediumcenter + HEV_km_largercenter*km_largercenter + HEV_km_hugecenter*km_hugecenter + HEV_Landuse*Landuse + HEV_NDVI*NDVI + HEV_km_super*km_super + HEV_km_station*km_station + HEV_km_bigstation*km_bigstation + HEV_km_bus*km_bus + HEV_Parkingspot*Parkingspot

#The utiltiy of a cartype is the constant (to replicate market shares) plus the utility from the fueltype and the utility from the weight-class. The latter are the same for the HEV (which is not in a seperate weight-class)
U_std_light = ASC_std_light + U_std + U_light
U_std_midlight = ASC_Std_midlight + U_std + U_midlight
U_std_midheavy = 0
U_std_heavy =  ASC_Std_heavy + U_std + U_heavy
U_diesel_midlight = ASC_Diesel_midlight + U_diesel + U_midlight
U_diesel_midheavy = ASC_Diesel_midheavy + U_diesel + U_midheavy
U_diesel_heavy = ASC_Diesel_heavy + U_diesel + U_heavy
U_HEV = ASC_HEV + U_HEV_plain
V1car = {11: U_std_light, 12: U_std_midlight, 13: U_std_midheavy, 14: U_std_heavy, 22: U_diesel_midlight, 23: U_diesel_midheavy, 24: U_diesel_heavy, 5: U_HEV} #Note: the first number gives the fuel and the second the weight (13 is fuel 1, namely standard and weight 3, namely midheavy)

#The utility of a fueltype or weight-class in a multicar household is the utility in a one-car household plus the extra utility of having that fueltype or weight-class in a multicar household.
U_std_2car = U_std + Dummy_2car_std
U_diesel_2car = U_diesel + Dummy_2car_diesel
U_light_2car = U_light + Dummy_2car_light
U_midlight_2car = U_midlight + Dummy_2car_midlight
U_midheavy_2car = U_midheavy + Dummy_2car_midheavy
U_heavy_2car = U_heavy + Dummy_2car_heavy
U_HEV_2car = U_HEV_plain + Dummy_2car_HEV

#The utility per cartype in a multicar household is constructed in the exact same way as in a onecar household
U2car_std_light = ASC_std_light + U_std_2car + U_light_2car
U2car_std_midlight = ASC_Std_midlight + U_std_2car + U_midlight_2car
U2car_std_midheavy = 0 
U2car_std_heavy =  ASC_Std_heavy + U_std_2car + U_heavy_2car
U2car_diesel_midlight = ASC_Diesel_midlight + U_diesel_2car + U_midlight_2car
U2car_diesel_midheavy = ASC_Diesel_midheavy + U_diesel_2car + U_midheavy_2car
U2car_diesel_heavy = ASC_Diesel_heavy + U_diesel_2car + U_heavy_2car
U2car_HEV = ASC_HEV + U_HEV_2car
V2car = {11: U2car_std_light, 12: U2car_std_midlight, 13: U2car_std_midheavy, 14: U2car_std_heavy, 22: U2car_diesel_midlight, 23: U2car_diesel_midheavy, 24: U2car_diesel_heavy, 5: U2car_HEV}

#The total likelihood is the of the likelihoods for one and two (plus) car households weighted by the probability of a household having one or two (plus) cars
latentprob =  (models.logit(V1car, None, Type)*Ponecar + models.logit(V2car, None, Type)*Pmorecars)**(CarValid == 1) #To the power of carvalid so that carless households will be excluded.
logprob =  log(latentprob*CarOwnProb) #Integrate with the first multinomial classification model
weight = HH_weight #Take into account the sample weights (which you rescaled above)
formulas = {'loglike': logprob, 'weight': weight}

biogeme = bio.BIOGEME(database, formulas) 
biogeme.generatePickle = False
biogeme.saveIterations= False
biogeme.modelName = 'Article2_InitialSubmission_FinalModel_20231020dataCleaning'

### Estimate the parameters
results = biogeme.estimate()
BioResults = results.getEstimatedParameters() 
BioSummary = results.shortSummary()
BioStats = results.getGeneralStatistics()

AIC = round(BioStats['Akaike Information Criterion'][0])

BioResultsshort = BioResults.round(3)
BioResultsshort = BioResultsshort[['Value', 'Std err','t-test','p-value']]
BioResultsshort.to_excel('20231020_BioResults_LC_MNL_HHIDuniqueScaler.xlsx')


'''XXXXX SAVING PARAMETERS FOR PREDICTIONS XXXXX'''

BioParameters = pd.DataFrame(BioResults['Value']).T
# BioParameters.to_excel('20230216_BioParameters_HHIDuniqueScaler.xlsx')
