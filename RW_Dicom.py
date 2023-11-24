
import pydicom as dm
import matplotlib.pyplot as plt 
import pandas as pd
import Find_ROI
import numpy as np 
import sys


file=sys.argv[1]
try:
    dcs=dm.dcmread(file)
    #dcs=dm.dcmread('FNXK0SN3\DNXK0TN2\I4000000')
    #dcs=dm.dcmread('FNXK0SN3\BNXK0QW0\I2400000')
    #dcs=dm.dcmread('28kV120mAs0cm')

    img=dcs.pixel_array

except:
    img=plt.imread(file)
    
plt.imshow(img)
plt.show()
#img = cv.imread('jpeg\image_s0023_i0001.jpg', cv.IMREAD_GRAYSCALE)
#img_pil=Image.fromarray(img)
#rotated_image1 = img_pil.rotate(180)
#rotated_image1.show()
#rot_npy=np.array(rotated_image1)
param=Find_ROI.find_ROIs(img,0.07, print_img=True)
#param=Find_ROI.find_ROIs(img,float(dcs[0x0018, 0x1164].value[0]), print_img=True)


# In[ ]:





# In[2]:


tags=[[0x0008,0x0020],[0x0008, 0x0060],[0x0008, 0x0070],[0x0008, 0x0080],[0x0018, 0x0060],[0x0018, 0x1150],[0x0018, 0x1152],[0x0018, 0x1160],[0x0018, 0x1191],[0x0018, 0x11a4],[0x0018, 0x7050]]
# colum=np.array([])
# for tag in tags:
#     colum=np.append(colum,dcs[tag].keyword)

# for k in param:
#     colum=np.append(colum, k)


# df=pd.DataFrame(columns=colum)
# print(df)

# df.to_csv('data',index=False)
   

# # px, py=dcs[0x0018, 0x1164].value
# # float(px)
# # 50/float(px)
# df


# In[4]:

print(param)

#df=pd.DataFrame()
#for tag in tags:
 #   df.loc[:,dcs[tag].keyword]=[dcs[tag].value]

#for k in param:
 #   df.loc[:,k]=[param[k]]

#df.to_csv('data',mode='a',index=False, header=False)


# In[6]:





# In[ ]:




