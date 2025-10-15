import pandas as pd
import numpy as np
from scipy.stats import truncnorm

def load_employee():
    df = pd.read_csv('data/employee.csv')
    df = df['Total Compensation']
    df = df.dropna().reset_index(drop=True) 
    data = df.to_numpy()
    return data, data.mean(), data.var()

def load_adult():
    df = pd.read_csv('data/adult.csv')
    df = df['age']
    df = df.dropna().reset_index(drop=True)
    data = df.to_numpy()
    return data, data.mean(), data.var()

def load_hpc_voltage():
    df = pd.read_csv('data/hpc.csv')
    df = df['Voltage'].astype(float)
    df = df.dropna().reset_index(drop=True)
    data = df.to_numpy()
    return data, data.mean(), data.var()

def data_load(dataset):
    if dataset == 'employee':
        return load_employee()
    elif dataset == 'adult':
        return load_adult()
    elif dataset == 'hpc_voltage':
        return load_hpc_voltage()
    else:
        raise ValueError(f'{dataset}: dataset config error.')