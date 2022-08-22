import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Expt_test.csv')

data_NF = pd.read_csv('Expt_testL.csv')

time = data['Time_Step'].values
r_vz = data['Wheelchair_z_velocity'].values

time_NF = data_NF['Time_Step'].values
r_vzNF = data_NF['Wheelchair_z_velocity'].values

plt.plot(time, r_vz)
plt.plot(time_NF, r_vzNF)
plt.show()
