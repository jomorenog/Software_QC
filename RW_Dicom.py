import pydicom as dm
import matplotlib.pyplot as plt 
import pandas as pd
import Find_ROI
import numpy as np 
import sys
import os.path

file=sys.argv[1]
try:
    path=sys.argv[2]
except:
    path='data.csv'

    
try:
    dcs=dm.dcmread(file)

    img=dcs.pixel_array

except:
    img=plt.imread(file)
    
param=Find_ROI.find_ROIs(img,0.07, print_img=True)

tags=[[0x0008,0x0020],[0x0008, 0x0060],[0x0008, 0x0070],[0x0008, 0x0080],[0x0018, 0x0060],[0x0018, 0x1150],[0x0018, 0x1152],[0x0018, 0x1160],[0x0018, 0x1191],[0x0018, 0x11a4],[0x0018, 0x7050]]

px, py=dcs[0x0018, 0x1164].value
param['MTF_50']=param['MTF_50']/float(px)
param['MTF_20']=param['MTF_20']/float(px)

#print(param)

df=pd.DataFrame()
colum=np.array([])
for tag in tags:
    colum=np.append(colum,dcs[tag].keyword)
    
for k in param:
    colum=np.append(colum, k)

df=pd.DataFrame(columns=colum)

for tag in tags:
    df.loc[:,dcs[tag].keyword]=[dcs[tag].value]

for k in param:
    df.loc[:,k]=[param[k]]
print(df)


if os.path.isfile(path):
    df.to_csv(path,mode='a',index=False, header=False)

else:
    df.to_csv(path,index=False)
    

