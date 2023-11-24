import numpy as np
import scipy.signal 
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import lfilter
from matplotlib.patches import Rectangle
from PIL import Image
import pydicom as dm
import MTF


def fun_profile(image, axes):
    
    '''
    Makes the projection over the selected axes
    '''
    
    if axes=='y':
        profile=np.zeros(image.shape[0])
        for i in range(image.shape[0]):
            profile[i]=np.mean(image[i,:])
    elif axes!='x':
        print('Error: invalid axes')
    else:
        profile=np.zeros(image.shape[1])
        for i in range(image.shape[1]):
            profile[i]=np.mean(image[:,i])
    
    # smooth the profile
    
    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
     
    return lfilter(b, a, profile)


def find_Cu(ar, eps):
    '''
    Search the limits of the Cu insert 
    '''
    flag=1 # If 1 the Cu is in the upper half 
           # If -1 the Cu insert is in the lower half 
    
    #Find the pixels where he intensity is near to the maximun 
    xx=np.where(abs(ar-np.max(ar))<eps*np.max(ar))
    hl_i, hl_f = [np.min(xx), np.max(xx)]
    xx=np.array(xx[0])
   
    #Remove the regions where the intensity is high but outside of the Cu ROI
    q1x, q2x, q3x = np.percentile(xx, [25, 50, 75])
    for x in xx:
        if abs(x-q2x)>1.2*(q3x-q1x):
            xx=np.delete(xx, np.where(xx==x)[0])
            
    return hl_i, xx[-1]


'''
def find_initial_al(ar, step, idx_dir ,initial_p):
    flag=1
    if idx_dir>len(ar/2):
        flag=-1
    #step=40
    index=initial_p+flag*step
    tem_max=ar[index]
    total_max=np.max(ar)

    #Salir del maximo
    if abs(tem_max-total_max)<0.1*total_max:
        index+=flag*step
        tem_max=w[index]
              
    #Salir del poso 
    sonda_stp=np.where(ar==tem_max)
    if tem_max<np.mean(ar[int(sonda_stp[0]):int(sonda_stp[0]+step)]):
        index+=flag*step
        tem_max=ar[index]
    
    in_idx=np.where(ar==tem_max)
    #plt.hlines(tem_max,0,4000, color='green')
    #plt.vlines(in_idx,0,1800, color='green')
    eps=0
    print('hola',index)
    while True:
        index+=step
        data=ar[int(in_idx[0]):index]
        tem_max=np.max(data)
        q1,q2,q3=np.percentile(ar[int(in_idx[0]):], [25, 50, 75])
        eps=1*(q3-q1)
        #print(eps)
        if (tem_max-q2)>eps:
        #index+=step
            plt.hlines([tem_max],0,4000, color='red')
            plt.vlines(index,0,1800, color='red')
            break
    
    ar_aux=ar[int(in_idx[0]):]
    xx=np.where(abs(ar_aux-tem_max)<0.01*tem_max)
    #print(xx)
    xx=np.array(xx[0])
    xx+=in_idx[0]
    #print(xx)
    q1x, q2x, q3x = np.percentile(xx, [25, 50, 75])
   # print(xx)
    for x in xx:
        if abs(x-q2x)>1.2*(q3x-q1x):
            xx=np.delete(xx, np.where(xx==x)[0])
            #print(x)
    while (xx[-1]-xx[0])>0.05*len(ar):
        xx=np.delete(xx, -1)
    plt.vlines(xx,0, 2000, color='black')
    hl_i, hl_f = [np.min(xx), np.max(xx)]
    
    
    print('hola',hl_i, hl_f)
    #plt.vlines(q3x,0, 2000, color='red')
    #plt.vlines(np.percentile(xx, 75),0, 2000, color='black')
    return hl_i, hl_f
'''
def find_Al(Img):
   
    # Find the initial profiles a identify the limits of the Cu region
    prof_x=fun_profile(Img, 'x')
    hiCu_x, hfCu_x=find_Cu(prof_x, 0.1)
    prof_y=fun_profile(Img, 'y')
    hiCu_y, hfCu_y=find_Cu(prof_y, 0.1)
    
    '''  
    Limit the image to the region that contain only the Al insert 
    if hfCu_y>0.6*len(prof_y):
        img_aux=Image.fromarray(Img)
        rotate_image=img_aux.rotate(180)
        new_image=np.array(rotate_image)
        row,col = new_image.shape
        im_red=new_image[row-hiCu_y:,col-hfCu_x:]
    else:
    '''     
    im_red=Img[hfCu_y:,int(2*hfCu_x):]
    
    
    #plt.imshow(im_red)
    #plt.show()
    im_red= cv.GaussianBlur(im_red, (5, 5), 10)
    p,im_thr=cv.threshold(im_red, np.percentile(im_red,98), 255, 0 )
    im_thr = cv.GaussianBlur(im_thr, (5, 5), 1)  # simple noise removal
    #plt.imshow(im_thr, cmap='gray')
    #plt.show()
    
    #Profile in axis x
    prof_x=fun_profile(im_thr,'x')
    sup_lim_x = len(prof_x)
    inf_lim_x=0
    plt.plot(prof_x)
    #plt.show()
    # Remove the region with low intensity in the edge of the image 
    
    if np.mean(prof_x[int(sup_lim_x-5):sup_lim_x])<np.percentile(prof_x,10):
        j=0
        while np.mean(prof_x[sup_lim_x-5:sup_lim_x])<np.percentile(prof_x,10):
            sup_lim_x=sup_lim_x-5
            j+=1
            if j>50:
                #print(len(prof_y)*(1-0.05))
                print('Error: It was not possible to remove the the region with high intensity outside of the Al ROI')
                break
   
    # Remove the region with high intensity in the edge of the image 
    i=0
    while np.mean(prof_x[int(sup_lim_x*(1-0.05)):sup_lim_x])>np.percentile(prof_x,85):
        sup_lim_x=int(sup_lim_x*(1-0.05))
        i+=1
        if i>50:
            #print(len(prof_y)*(1-0.05))
            print('Error: It was not possible to remove the the region with high intensity outside of the Al ROI')
            break
    #plt.plot(prof_x[:sup_lim_x])
    #plt.hlines(np.percentile(prof_x,95), 0, 2000)
    #plt.show()
    #Find Al region
    
    thre_x=np.percentile(prof_x[:sup_lim_x],95)
    al_x=np.where(prof_x[:sup_lim_x]>thre_x)

   # plt.vlines(al_x, 0, 100)
   # plt.show()
    
    #Profile in axis y
    prof_y=fun_profile(im_thr,'y')
    sup_lim_y=len(prof_y)
    #plt.plot(prof_y)
    
    # Remove the region with low intensity in the edge of the image 
    if np.mean(prof_y[int(sup_lim_y-5):sup_lim_y])<np.percentile(prof_y,30):
        j=0
        while np.mean(prof_y[sup_lim_y-5:sup_lim_y])<np.percentile(prof_y,30):
            sup_lim_y=sup_lim_y-5
            j+=1
            if j>50:
                #print(len(prof_y)*(1-0.05))
                print('Error: It was not possible to remove the the region with high intensity outside of the Al ROI')
                break
                
    
    # Remove the region with high intensity in the edge of the image 
    while np.mean(prof_y[int(sup_lim_y*(1-0.05)):sup_lim_y])>np.percentile(prof_y,85):
        sup_lim_y=int(sup_lim_y*(1-0.05))
        i+=1
        if i>10:
            print(len(prof_y)*(1-0.05))
            break
    
    #plt.plot(prof_y[:sup_lim_y])
    #plt.show()
    thre_y=np.percentile(prof_y[:sup_lim_y],95)
    al_y=np.where(prof_y[:sup_lim_y]>thre_y)
   
    return [al_x[0][0]+2*hfCu_x,al_x[0][-1]+2*hfCu_x], [al_y[0][0]+hfCu_y,al_y[0][-1]+hfCu_y]

    
def find_center(arImg,arCu, arAl):
    # Search the background's ROI from the Cu and Al ROI 
    sup_lim=0
    inf_lim=0
    
    if arCu[0]>len(arImg)/2:
        sup_lim=arCu[0]
        inf_lim=arAl[1]
    else:
        sup_lim=arAl[0]
        inf_lim=arCu[1]
        
    return inf_lim, sup_lim


def find_ROIs(image, px_size ,print_img=False):


    prof_x=fun_profile(image, 'x')
#     plt.plot(prof_x,'k')
#     plt.plot(prof_x[10:750],'r')
#     plt.plot(np.arange(2700,2950),prof_x[2700:2950],'r')
#     plt.plot(np.arange(3050,3310),prof_x[3050:3310],'r')
#     plt.arrow(3180,1100,0,-300, head_width=50, color='k')
#     plt.text(2800,500,'Etiqueta \nde imagen', fontsize=14)
#     plt.text(320,1200,'Cu', fontsize=16, color='r')
#     plt.text(2750,1300,'Al', fontsize=16, color='r')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlabel('Pixel',fontsize=14)
#     plt.ylabel('Intensidad',fontsize=14)
#     #plt.savefig('perfil_señalado.pdf')
#     plt.show()
    hiCu_x, hfCu_x=find_Cu(prof_x, 0.1)
    #'''

    if hfCu_x>0.6*len(prof_x):
        img_aux=Image.fromarray(image)
        rotate_image=img_aux.rotate(180)
        new_image=np.array(rotate_image)
        image = new_image
    else:
        image=image


    prof_x=fun_profile(image, 'x')
    hiCu_x, hfCu_x=find_Cu(prof_x, 0.1)

    
    prof_y=fun_profile(image, 'y')
    hiCu_y, hfCu_y=find_Cu(prof_y, 0.1)
    
    hAl_x, hAl_y=find_Al(image)
    hiAl_x, hfAl_x=hAl_x[0],hAl_x[1]
    hiAl_y, hfAl_y=hAl_y[0],hAl_y[1]

    icenter_x,fcenter_x=find_center(prof_x,[hiCu_x, hfCu_x],[hiAl_x, hfAl_x])
    icenter_y,fcenter_y=find_center(prof_y,[hiCu_y, hfCu_y],[hiAl_y, hfAl_y])
    
    xCu=(hfCu_x-hiCu_x)//3
    yCu=(hfCu_y-hiCu_y)//3

    ecor=1

    xAl=(hfAl_x-hiAl_x)//2
    yAl=(hfAl_y-hiAl_y)//2
    
    #Al ROI
    w_al=5/px_size
    
    #Bg ROI
    w_bg=40/px_size
    
    
    #Cu ROI
    xw_Cu=30/px_size
    yw_Cu=15/px_size
    
    
    if print_img:
        plt.imshow(image, cmap='gray')
        ax = plt.gca()

        # Create a Rectangle patch
        rect_Cu = Rectangle((hfCu_x-xCu,hiCu_y+yCu),xw_Cu,yw_Cu,linewidth=1,edgecolor='r',facecolor='none')
        rect_Al = Rectangle((hiAl_x+xAl/3,hiAl_y+yAl/3),w_al,w_al,linewidth=1,edgecolor='g',facecolor='none')
        rect_center=Rectangle((icenter_x+(fcenter_x-icenter_x)*(1/2-1/8),icenter_y+(fcenter_y-icenter_y)*(1/2-1/8)),w_bg,w_bg,linewidth=1,edgecolor='b',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect_Cu)
        ax.add_patch(rect_Al)
        ax.add_patch(rect_center)
        plt.xticks([])
        plt.yticks([])
       # plt.savefig('Image_phantom.pdf')
        plt.show()
        
    #Measuring
    
    mtf_x_roi=image[hiCu_y+yCu:hiCu_y+yCu+int(yw_Cu), hfCu_x-xCu:hfCu_x-xCu+int(xw_Cu)]
#     plt.imshow(mtf_x_roi)
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig('cu.pdf')
#     plt.show()
    bg_roi=image[icenter_y+int((fcenter_y-icenter_y)*(1/2-1/8)):icenter_y+int((fcenter_y-icenter_y)*(1/2-1/8)+w_bg), icenter_x+int((fcenter_x-icenter_x)*(1/2-1/8)):icenter_x+int((fcenter_x-icenter_x)*(1/2-1/8)+w_bg)]
#     plt.imshow(bg_roi)
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig('bg.pdf')
#     plt.show()
    al_roi=image[hiAl_y+yAl//3:hiAl_y+yAl//3+int(w_al),hiAl_x+xAl//3:hiAl_x+xAl//3+int(w_al)]
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(al_roi)
#     plt.savefig('al.pdf')
#     plt.show()
    
    
    MTF_50, MTF_20, MTF_Ny, x,mtf=MTF.measure_MTF(mtf_x_roi)
    #print(mtf_x_roi[0])
    SNR_bg=np.mean(bg_roi)/np.std(bg_roi)
    SNR_al=np.mean(al_roi)/np.std(al_roi)
    SDNR=abs(np.mean(al_roi)-np.mean(bg_roi))/np.std(bg_roi)
    
    IMAGE_PARAM={
        'MTF_50':MTF_50,
        'MTF_20':MTF_20,
        'MTF_Ny':MTF_Ny,
        'SNR_bg':SNR_bg,
        'SNR_al':SNR_al,
        'SDNR':SDNR
    }
    
    
    return IMAGE_PARAM



