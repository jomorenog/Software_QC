#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
import statsmodels.api as sm 
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

def gauss(x,A,mu,sigma):
    return A*np.exp(-0.5*((x-mu)/sigma)**2)

def fermi(x,a0,a1,c1,c2):
    return a0+a1*(1/(1+np.exp(c1*(x-c2))))


# In[8]:



angle=0

def Get_ESF(image):
    #img= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img=image
    
    
    
    img= cv.GaussianBlur(img, (3, 3), 0)  # simple noise removal
    p,im_thr=cv.threshold(img, 0.6*np.max(img), np.max(img), cv.THRESH_BINARY)
    img_c=np.uint8(im_thr)
    #img_c=im_thr
 
    
    #img_c= cv.GaussianBlur(img_c, (3, 3), 0)  # simple noise removal
    
    
    #img_c = cv.GaussianBlur(im_thr, (5, 5), 1)  # simple noise removal
    #img_c= cv.GaussianBlur(img_c, (7, 7), 1) 
    edges = cv.Canny(img_c,0.2*np.max(img_c),0.8*np.max(img_c))
    parameters=np.polyfit(*reversed(np.nonzero(edges)), deg=1)
    plt.imshow(img_c)
    plt.colorbar()
    plt.show()
    line=np.nonzero(edges)
    y=line[0][-1]-line[0][0]
    x=line[1][0]-line[1][-1]

    angle=np.arctan(x/y)

    z=np.array([])
    l=np.array([])

    rows, clms=(img.shape)
    for i in range(rows):
        for idx, pxl in enumerate(img[i:i+1, :][0]):
            z=np.append(z,(idx-(i-parameters[1])/parameters[0])*np.cos(angle))
            l=np.append(l,pxl)
       
    z_lim=np.min([abs(z[0]),abs(z[-1])])
    z_range=z_lim//3
    z_sample=np.arange(-3*z_range,3*z_range, 3)


    mean=np.zeros(clms)
    for i in range(clms):
        mean[i]=np.mean(img[:,i])

    z, l = zip(*sorted(zip(z, l)))
    z=np.array(z)
    l=np.array(l)
    intr=np.interp(z_sample, z,l)

    print(np.rad2deg(angle))
    
    n=1#np.cos(angle)
    z_ran=(z[-1]-z[0])//n
    z_max=n*z_ran


    l_m=np.array([])

    for i in range(int(z_ran)):
        l_m=np.append(l_m,np.mean(l[np.where(abs(z-(z[0]+n/2+i*n))<n/2)[0]]))
        
   # plt.plot(np.arange(z[0], z[0]+z_max, n), l_m, 'ko')
    popt, pcov=curve_fit(fermi, np.arange(z[0], z[0]+z_max, n),l_m,[np.min(l_m),np.max(l_m),1, 0])
    #plt.plot(np.arange(z[0], z[0]+z_max,0.1*n), fermi(np.arange(z[0], z[0]+z_max, 0.1*n),*popt), 'r-')
    
#     plt.xlabel('subpixel',fontsize=14)
#     plt.ylabel('ESF',fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
    #plt.show()
    #xval=np.arange(-10, 10,np.cos(angle)/np.cos(angle))
    x_val=np.arange(z[0], z[0]+z_max, 1*n)
    return x_val,z, popt
    
    
def Get_LSF(x_val, popt):
    xval=np.arange(-10, 10,np.cos(angle)/np.cos(angle))
    lsf=np.gradient(fermi(xval,*popt),xval)
    lsf=np.absolute(lsf)
    #plt.plot(xval,np.absolute(lsf),'ko')
    
    # lsf=np.gradient(intr ,xval)
    # lsf_f= lfilter(b, a, lsf)
    # #plt.plot(sam[:,0], sam[:,1])
    # plt.plot(xval,np.absolute(lsf),'ro')
    # #plt.plot(z,w,'g-')
    # plt.plot(xval,lsf_f,'g-')

    popt_g, pcov_g= curve_fit(gauss, xval, lsf,[np.max(lsf),0,1] )
    lsf=gauss(xval, *popt_g)
#     plt.plot(np.arange(z[0], z[0]+z_max, 0.1*n),gauss(np.arange(z[0], z[0]+z_max, 0.1*n),*popt_g),'r-')
#     plt.xlim(-10,10)
#     plt.xlabel('subpixel',fontsize=14)
#     plt.ylabel('LSF',fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     #plt.savefig('lsf.pdf')
    return lsf


def Get_MTF(x_val, lsf,print_image=False):
    #n=4#np.cos(angle)
    #z_ran=(z[-1]-z[0])//n
    #z_max=n*z_ran
    #px=1/(1*(x_val[len(x_val)//2]-x_val[0]))
    #N=len(lsf)
    #T=0.25
    #fft1=np.abs(fft(lsf[:]))
    fft1_g=np.abs(fft(lsf))
    #fft1_g=np.abs(fft(gauss(x_val,*popt_g)))
    #N_g=len(fft1_g)
    #xf = fftfreq(N, T)[:N//2]
    #x_mtf=1*np.arange(0, N)*px
    #x_mtf_g=1*np.arange(0, N_g)*px
    mtf=abs(fft1_g[:len(fft1_g)//2]/fft1_g[0])
    x_mtf=np.arange(0,len(mtf))/len(mtf)
    MTF_50=np.interp(0.5,mtf[::-1],x_mtf[::-1])
    MTF_20=np.interp(0.2,mtf[::-1],x_mtf[::-1])
    MTF_Ny=np.interp(0.5,x_mtf,mtf)
    
    if print_image:
        plt.plot(x_mtf,mtf,'k')
        # for idx, f in enumerate(x_mtf_g[:len(x_mtf_g)//2]):
        #     if f==0:
        #         fft1_g[:len(fft1_g)//2][idx]=fft1_g[:len(fft1_g)//2][idx]
        #     else:    
        #         fft1_g[:len(fft1_g)//2][idx]=fft1_g[:len(fft1_g)//2][idx]*((np.pi*f*0.5)/(np.sin(np.pi*f*0.5)))

        #plt.plot(x_mtf_g[:len(fft1_g)//2],fft1_g[:len(fft1_g)//2]/fft1_g[0])

    #     print(len(fft1), len(xval))
    #     print(np.sinc(np.pi/4))
    #     print((np.pi*0.8*0.5)/(np.sin(np.pi*0.5*0.8)))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('Ciclos/pixel', fontsize=14)
        plt.ylabel('MTF', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
       
        plt.grid()
        plt.hlines(0.5,0,MTF_50, linestyles='dashed')
        plt.vlines(MTF_50, 0, 0.5, linestyles='dashed')
        plt.hlines(0.2,0,MTF_20, linestyles='dashed',color='r')
        plt.vlines(MTF_20, 0, 0.2, linestyles='dashed',color='r')
        plt.vlines(0.5, 0, MTF_Ny, linestyles='dashed',color='g')
        plt.text(0.5, MTF_Ny+0.1,'Frecuencia Nyquist', fontsize=14, color='g')
        plt.arrow(0.5, MTF_Ny,0.05,0.07,width=0.005,color='g')
        #plt.savefig('MTF_image.pdf')

    return MTF_50, MTF_20, MTF_Ny, x_mtf, mtf
    

    
def measure_MTF(image, plot=False):
    xx, z,par=Get_ESF(image)
    lsf=Get_LSF(xx, par)
    m_50, m_20, m_Ny,x, m=Get_MTF(xx, lsf,plot)
    return m_50, m_20, m_Ny,xx, m

