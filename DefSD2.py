
# coding: utf-8

# In[1]:
  
import numpy as np   #Herramientas paa manejar arreglos multidimensionales
import matplotlib.pyplot as plt   #Gráficos, histogramas, gráfico de dispersión
from astropy import coordinates as coords   #Conversion, sistemas y marcos de referencia
from astropy import units as u   #Conversion y desarrollo de operaciones aritméticas de instancias Quantity
from astropy.io import fits   
from astropy.table import Table, Column ,vstack
from astropy.modeling import models, fitting   #Representación y ajuste de modelos en 1D y 2D
from scipy.integrate import quad
import pyneb as pn
import random

# In[ ]:

gaussian = lambda x,a,b,c : a * np.exp(-0.5* (x - b)**2 / c**2)
diags = pn.Diagnostics()

# In[ ]:

def FWHM(X,Y):
    half_max = (max(Y)+min(Y)) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #plot(X,d) #if you are interested
    #find the left and right most indexes
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return X[right_idx] - X[left_idx] #return the difference (full width)


# In[ ]:

def lineProfile(i,spec, lambd,ListLines,ListGal):
    linePr = Table(names=('lambda', 'inf', 'sup'), dtype=('i5','i5', 'i5'))
    for c in range(0, len(ListLines)):
        lambdLine=ListLines['LAMBDA VAC ANG'][c]          #Líneas obtenidas de la base de datos Astroquery
        v = np.where((lambd >= lambdLine-1) & (lambd <= lambdLine+1))
        ind = 1
        f = FWHM(lambd[v[0][0]-ind:v[0][0]+ind], ListGal[i]['flux'][v[0][0]-ind:v[0][0]+ind])
        while (len(f)==0):
            ind = ind+1
            f = FWHM(lambd[v[0][0]-ind:v[0][0]+ind], ListGal[i]['flux'][v[0][0]-ind:v[0][0]+ind])
        l_inf = v[0][0]-ind
        l_sup = v[0][0]+ind
        # Una vez encontrado el intervalo se busca el indice donde
        # tiene el máximo valor de intensidad
        indMaxInt = np.where(ListGal[i]['flux'][l_inf:l_sup]==np.max(ListGal[i]['flux'][l_inf:l_sup]))
        indMax=indMaxInt[0][0]+l_inf
        iM = indMax
        l_inf = 1
        l_sup = 1
        a=ListGal[i]['flux'][indMax]
        while (ListGal[i]['flux'][indMax-l_inf] <= a) | ((ListGal[i]['flux'][indMax-l_inf]-spec[indMax-l_inf])>10):
            a = ListGal[i]['flux'][indMax-l_inf]
            l_inf = l_inf+1

        a=ListGal[i]['flux'][indMax]
        while (ListGal[i]['flux'][indMax+l_sup] <= a) | ((ListGal[i]['flux'][indMax+l_sup]-spec[indMax+l_sup])>10):
            a = ListGal[i]['flux'][indMax+l_sup]
            l_sup = l_sup+1

        l_inf = indMax-l_inf+1
        l_sup = indMax+l_sup
        linePr.add_row((iM, l_inf, l_sup))        
    return linePr


# In[ ]:

specSN = []
lambdSN = []

specP = []
lambdP = []

specD = []
lambdD = []
def r(spec,lambd):
    for b in range(0,len(spec)-1,1):
        pend=(spec[b+1]-spec[b])/(lambd[b+1]-lambd[b])
        specP.append(np.fabs(pend))
        lambdP.append(lambd[b])

        specD.append(np.fabs(spec[b+1]-spec[b]))
        lambdD.append(lambd[b])
        cont = 0
        m=spec[0]
        if b>0:
            m= np.mean(spec[0:b])
        if b>=30:
            m= np.mean(spec[b-30:b-20])
        if (np.fabs(pend)<1):
            specSN.append(spec[b])
            lambdSN.append(lambd[b])
        while (np.fabs(pend)>0.3):
            cont = cont +1
            if spec[b]>m:
                if (spec[b]<spec[b+1]):
                    spec[b+1]=(spec[b+1]+ spec[b])/2 
                else:
                    spec[b]=(spec[b+1]+ spec[b])/2     
            if spec[b]<=m:
                if (spec[b]<spec[b+1]):
                    spec[b]=(spec[b+1]+ spec[b])/2  
                else:
                    spec[b+1]=(spec[b+1]+ spec[b])/2   
            pend=(spec[b+1]-spec[b])/(lambd[b+1]-lambd[b])

    f=len(spec)
    for b in range(1,len(spec)-1,1):
        v=f-b
        pend=(spec[v-1]-spec[v])/(lambd[v-1]-lambd[v])
        m=spec[f-1]
        if v<(f-1):
            m= np.mean(spec[v:f-1])
        if v<(f-30):
            m= np.mean(spec[v+20:v+30])
        while (np.fabs(pend)>0.15):
            if spec[v]>m:
                if (spec[v]<spec[v-1]):
                    spec[v-1]=(spec[v-1]+ spec[v])/2    
                else:
                    spec[v]=(spec[v-1]+ spec[v])/2    
            if spec[v]<=m:
                if (spec[v]<spec[v-1]):
                    spec[v]=(spec[v-1]+ spec[v])/2 
                else:
                    spec[v-1]=(spec[v-1]+ spec[v])/2  
            pend=(spec[v-1]-spec[v])/(lambd[v-1]-lambd[v])
    return spec,lambd


# In[ ]:

def finalFit(i, spec,lambd,ListLines,ListGal,iterr):
    lineP = lineProfile(i,spec,lambd,ListLines,ListGal)
    f1 = fitting.LevMarLSQFitter()
    Gaus=[]
    lineStdDev = 3.5
    for x in range(len(ListLines)):
        lineAmplitude = ListGal[i]['flux'][lineP['lambda'][x]]
        v = np.where((lambd >= ListLines['LAMBDA VAC ANG'][x]-1) & (lambd <= ListLines['LAMBDA VAC ANG'][x]+1))
        ampMax= ListGal[i]['flux'][v[0][0]]
        Gaus.append(models.Gaussian1D(amplitude=lineAmplitude,mean=ListLines['LAMBDA VAC ANG'][x],stddev=lineStdDev))#,
                                      #bounds={'amplitude':(0, ampMax),'mean':(ListLines['LAMBDA VAC ANG'][x]-1.0, ListLines['LAMBDA VAC ANG'][x]+1.0),'stddev':(0.5, 22.5)}))
    
    sum_Gaussian=Gaus[0]+Gaus[1]+Gaus[2]+Gaus[3]+Gaus[4]+Gaus[5]+Gaus[6]+Gaus[7]+Gaus[8]+Gaus[9]+Gaus[10]+Gaus[11]+Gaus[12]
    sum_Ga2=Gaus[0]+Gaus[1]
    if len(ListLines) >= 14:
		sum_Gaussian=Gaus[0]+Gaus[1]+Gaus[2]+Gaus[3]+Gaus[4]+Gaus[5]+Gaus[6]+Gaus[7]+Gaus[8]+Gaus[9]+Gaus[10]+Gaus[11]+Gaus[12]+Gaus[13]+Gaus[14]
    gaussian_fit = f1(sum_Gaussian, lambd, ListGal[i]['flux']-spec, maxiter=iterr)
    gaussian_fit2 = f1(sum_Ga2, lambd, ListGal[i]['flux']  -spec, maxiter=iterr) 
    Graph=[]
    Graph.append(gaussian_fit)   
    Graph.append(gaussian_fit2)
    return Graph


# In[2]:

#---->   http://www.stecf.org/software/ASTROsoft/DER_SNR/der_snr.py
# =====================================================================================

def DER_SNR(flux):
   
# =====================================================================================
   """
   DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
               definition set forth by the Spectr2al Container Working Group of ST-ECF,
	       MAST and CADC. 

               signal = median(flux)      
               noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
	       snr    = signal / noise
               values with padded zeros are skipped

   USAGE       snr = DER_SNR(flux)
   PARAMETERS  none
   INPUT       flux (the computation is unit independent)
   OUTPUT      the estimated signal-to-noise ratio [dimensionless]
   USES        numpy      
   NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum 
	       as a whole as long as
               * the noise is uncorrelated in wavelength bins spaced two pixels apart
               * the noise is Normal distributed
               * for large wavelength regions, the signal over the scale of 5 or
	         more pixels can be approximated by a straight line
 
               For most spectr2a, these conditions are met.

   REFERENCES  * ST-ECF Newsletter, Issue #42:
               www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
               * Software:
	       www.stecf.org/software/ASTROsoft/DER_SNR/
   AUTHOR      Felix Stoehr, ST-ECF
               24.05.2007, fst, initial import
               01.01.2007, fst, added more help text
               28.04.2010, fst, return value is a float now instead of a numpy.float64
   """
   from numpy import array, where, median, abs 

   flux = array(flux)

   # Values that are exactly zero (padded) are skipped
   flux = array(flux[where(flux != 0.0)])
   n    = len(flux)      

   # For spectr2a shorter than this, no value can be returned
   if (n>4):
      signal = median(flux)

      noise  = 0.6052697 * median(abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))

      return float(signal / noise)  

   else:

      return 0.0

# end DER_SNR -------------------------------------------------------------------------

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

rc = pn.RedCorr()
rc.law = 'G03 LMC'
IntrinsicHB=np.linspace(2.8,3.1,31)
