
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
from scipy.integrate import simps
import sys
import traceback
import math as mt


# In[ ]:

#   ----------------------------------------------------------------------------------------------------------------------
gaussian = lambda x,a,b,c : a * np.exp(-0.5* (x - b)**2 / c**2)
diags = pn.Diagnostics()

# In[ ]:

#   ----------------------------------------------------------------------------------------------------------------------
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

#   ----------------------------------------------------------------------------------------------------------------------
#   a function that obtain the slope and the intersection given two points within it

def linModel(x0, y0, x1, y1):
    m=(y1-y0)/(x1-x0)
    b=y1-m*x1
    return m, b

#   linear function with an np.array : x, a slope : a and intersection : c 

def lines(x, a, c):
    return a*x + c

lorentzFunc = lambda x,A,f,x0 : (A*f**2)/(f**2 + (x-x0)**2)
#def lorentzFunc(lal, A, f, x0):
#    return (A*f**2)/(f**2 + (x-x0)**2)

#   ----------------------------------------------------------------------------------------------------------------------
def fit2(i, spec,lambd,ListLines,ListGal,iterr):
    lineP = lineProfile(i,spec,lambd,ListLines,ListGal)
    f1 = fitting.LevMarLSQFitter()
    Lorentz = []
    lineStdDev = 0.5 #3.5
    for x in range(len(ListLines)):
        lineAmplitude = ListGal[i]['flux'][lineP['lambda'][x]]
        v = np.where((lambd >= ListLines['LAMBDA VAC ANG'][x]-1) & (lambd <= ListLines['LAMBDA VAC ANG'][x]+1))
        ampMax= ListGal[i]['flux'][v[0][0]]
        Lorentz.append(models.Lorentz1D(amplitude=lineAmplitude,x_0=ListLines['LAMBDA VAC ANG'][x],fwhm=2.355*lineStdDev, bounds={'amplitude':(0, ampMax)}))
    
    sum_Lorentz=Lorentz[0]+Lorentz[1]+Lorentz[2]+Lorentz[3]+Lorentz[4]+Lorentz[5]+Lorentz[6]+Lorentz[7]+Lorentz[8]+Lorentz[9]+Lorentz[10]+Lorentz[11]+Lorentz[12]
    sum_Lo2=Lorentz[0]+Lorentz[1]
    if len(ListLines) >= 14:
        sum_Lorentz=Lorentz[0]+Lorentz[1]+Lorentz[2]+Lorentz[3]+Lorentz[4]+Lorentz[5]+Lorentz[6]+Lorentz[7]+Lorentz[8]+Lorentz[9]+Lorentz[10]+Lorentz[11]+Lorentz[12]+Lorentz[13]+Lorentz[14]
    lorentz_fit = f1(sum_Lorentz, lambd, ListGal[i]['flux']-spec, maxiter=iterr)
    lorentz_fit2 = f1(sum_Lo2, lambd, ListGal[i]['flux']  -spec, maxiter=iterr) 
    LOR=[]
    LOR.append(lorentz_fit)   
    LOR.append(lorentz_fit2)
    return LOR

#   ----------------------------------------------------------------------------------------------------------------------
#   Obtain the index (Table linePr) of the feet line profile given a index of the galaxy : i, 
#   the continuum spectrum np.array : spec, the wavelenghts : lambd, data of the lines : ListLines,
#   the list of spectrum galaxies : ListGal, and minimum resolution of the spectrum : limResoluc

#   Aqui encontré muchos problemas porque cuando se encuentran las lineas acopladas de OII en 3726-29
#   se debe encontrar los límites de los parámetros

def lineProfile(i,spec, lambd,ListLines,ListGal, limResoluc):
    linePr = Table(names=('lambda', 'inf', 'sup'), dtype=('i5','i5', 'i5'))
    for c in range(0, len(ListLines)):
        lambdLine=ListLines['LAMBDA VAC ANG'][c]          #Líneas obtenidas de la base de datos Astroquery
        v = np.where((lambd >= lambdLine-limResoluc) & (lambd <= lambdLine+limResoluc))
        if len(v[0]) == 0:
            ind = 0
            l_inf = 0
            l_sup = 0
            iM=0
        else:
            ind = 1
            try:
                f = FWHM(lambd[v[0][0]-ind:v[0][0]+ind], ListGal[i][v[0][0]-ind:v[0][0]+ind])
            except IndexError:
                f=[]
            while (len(f)==0):
                ind = ind+1
                try:
                    f = FWHM(lambd[v[0][0]-ind:v[0][0]+ind], ListGal[i][v[0][0]-ind:v[0][0]+ind])
                except IndexError:   
                    ind = ind+1
                    f=[]
            l_inf = v[0][0]-ind
            l_sup = v[0][0]+ind
            # Una vez encontrado el intervalo se busca el indice donde
            # tiene el máximo valor de intensidad
            indMaxInt = np.where(ListGal[i][l_inf:l_sup]==np.max(ListGal[i][l_inf:l_sup]))
    #        print v
            indMax=indMaxInt[0][0]+l_inf
            iM = indMax
            l_inf = 1
            l_sup = 1
            a=ListGal[i][indMax]
            while (ListGal[i][indMax-l_inf] <= a):  # | ((ListGal[i][indMax-l_inf]-spec[indMax-l_inf])>10):
                a = ListGal[i][indMax-l_inf]
                l_inf = l_inf+1

            a=ListGal[i][indMax]
            while (ListGal[i][indMax+l_sup] <= a):  # | ((ListGal[i][indMax+l_sup]-spec[indMax+l_sup])>10):
                a = ListGal[i][indMax+l_sup]
                l_sup = l_sup+1

            l_inf = indMax-l_inf+1
            l_sup = indMax+l_sup
        linePr.add_row((iM, l_inf, l_sup))        
    return linePr


# In[ ]:

#   ----------------------------------------------------------------------------------------------------------------------
# Esta función se llama cuando la amplitud encontrada es negativa antes de hacer el ajuste (intensidad observada - continuo)
# entonces se calcula nuevamente el continuo en esta linea y alrededores para hacer el ajuste.

#lineContCorrection(lambd, spec, x, lineP, ListGal, i)

def lineContCorrection(lambdLineRe, fluxLineRe, indexLineProf, tableLineProf, spectraF, indexG):
    medLineRe = tableLineProf[indexLineProf]['inf']+(tableLineProf[indexLineProf]['sup']-tableLineProf[indexLineProf]['inf'])/2
    minFluxLineRe = np.min(spectraF[indexG][tableLineProf[indexLineProf]['inf']:tableLineProf[indexLineProf]['sup']])
    max = len(lambdLineRe)-1
    for iarr in range(1, 20):
        indexL = medLineRe - iarr
        indexS = medLineRe + iarr-1
        if indexL>=0 :
            fluxLineRe[indexL] = (minFluxLineRe + fluxLineRe[indexL])/2
        if indexS<=max :
            fluxLineRe[indexS] = (minFluxLineRe + fluxLineRe[indexS])/2
    return fluxLineRe


#   ----------------------------------------------------------------------------------------------------------------------
#   obtain the continuum of the spectrum given a the observed spec : spectrum, wavelenght : lambd, and
#   a parameter of how variable is the continuum : ppend) 

def r(spec,lambd, ppend):
    for b in range(0,len(spec)-1,1):
        pend=(spec[b+1]-spec[b])/(lambd[b+1]-lambd[b])
        
        cont = 0
        m=spec[0]
        if b>0:
            m= np.mean(spec[0:b])
        if b>=30:
            m= np.mean(spec[b-30:b-20])
#        if (np.fabs(pend)<1):
#            specSN.append(spec[b])
#            lambdSN.append(lambd[b])
        while (np.fabs(pend)>ppend):
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
        while (np.fabs(pend)>ppend/2):
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

#   ----------------------------------------------------------------------------------------------------------------------
#  finalFit(index int :i, continuum spectrum np.array :spec, 
#      wavelengths np.array :lambd, wavelenghts of lines Astropy.Table : ListLines,
#      array of galaxies list :ListGal, error bar list : spIvar, maximum iteration Int :iterr,
#      resolution limit float :limResol):)

def finalFit(i, spec,lambd,ListLines,ListGal,spIvar,iterr, limResol):
    lineP = lineProfile(i,spec,lambd,ListLines,ListGal, limResol)   #  Table of line profile indexes upper and lower 
    linePaux=lineP['lambda'].data     
    alslk = lineP['lambda'][0]
         #    it obtain the number of lines overlayed in a profile
    for c in range(1,len(linePaux)):
        if (linePaux[c]==alslk) & (alslk!=0):
            linePaux[c-1] = 2
            linePaux[c] = 2
        else:
            linePaux[c] = 1
        alslk = lineP['lambda'][c]
        
    f1 = fitting.LevMarLSQFitter()
    lineStdDev = 3.5    #   default standard deviation
    condit = True #   esto es mejor quitarlo
    while condit:    #   esto es mejor quitarlo
        Gaus=[]      #    array of models
        for x in range(len(ListLines)):       #     for each line profile it is created a gaussian model 
            if lineP['inf'][x]==0:            
                Gaus.append(models.Gaussian1D(amplitude=0,mean=ListLines['LAMBDA VAC ANG'][x],stddev=0))
                      #    it didn't model a profile for this line
            else:
                v = np.where((lambd >= ListLines['LAMBDA VAC ANG'][x]-limResol) & (lambd <= ListLines['LAMBDA VAC ANG'][x]+limResol))
                      #    estimate the wavelenghts for the center of the line
                
                ampMax= ListGal[i][v[0][0]]-spec[v[0][0]]   #   max amplitude of the profile
                nLines = linePaux[x]                        #   obtain the number of overlaying line profiles 
                f = FWHM(lambd[lineP['inf'][x]:lineP['sup'][x]], ListGal[i][lineP['inf'][x]:lineP['sup'][x]])
                      #    calculate the FWHM of the profile
                coasi = 0
                while len(f)==0:
                    coasi = coasi + 1
                    f = FWHM(lambd[(lineP['inf'][x]-coasi):(lineP['sup'][x]+coasi)], ListGal[i][(lineP['inf'][x]-coasi):(lineP['sup'][x]+coasi)])
                           #  calculate FWHM so often as it is found a non profile spectrum
                        
                stdLine = (1/2.3548)*f[0]/nLines     #   estimate the observed standard deviation
                lineWidthBase = (lambd[lineP['sup'][x]] - lambd[lineP['inf'][x]])/(2*nLines)
                      ##   calculate the observed line width at the base of the profile
                if (ampMax>0) & (ampMax>3*spIvar[lineP['lambda'][x]]):  # & (coasi < 2):
                      #   a model for the observed amplitude (emission line) is greater than three times
                        #    the error of the intensity in the spectrum and the bounds depend on the number of lines
                        #   in the profile and the width of his base
                    Gaus.append(models.Gaussian1D(amplitude=ampMax,mean=ListLines['LAMBDA VAC ANG'][x],stddev=stdLine, bounds={'mean':(ListLines['LAMBDA VAC ANG'][x]-lineWidthBase/(2*nLines), ListLines['LAMBDA VAC ANG'][x]+lineWidthBase/(2*nLines)),'stddev':(0.8*stdLine, 1.5*stdLine)}))
                if ampMax <=0:
                    spec = lineContCorrection(lambd, spec, x, lineP, ListGal, i)
                    v = np.where((lambd >= ListLines['LAMBDA VAC ANG'][x]-limResol) & (lambd <= ListLines['LAMBDA VAC ANG'][x]+limResol))
                      #    estimate the wavelenghts for the center of the line
                    ampMax= ListGal[i][v[0][0]]-spec[v[0][0]]   #   max amplitude of the profile
                    nLines = linePaux[x]                        #   obtain the number of overlaying line profiles 
                    f = FWHM(lambd[lineP['inf'][x]:lineP['sup'][x]], ListGal[i][lineP['inf'][x]:lineP['sup'][x]])
                          #    calculate the FWHM of the profile
                    coasi = 0
                    while len(f)==0:
                        coasi = coasi + 1
                        f = FWHM(lambd[(lineP['inf'][x]-coasi):(lineP['sup'][x]+coasi)], ListGal[i][(lineP['inf'][x]-coasi):(lineP['sup'][x]+coasi)])
                           #  calculate FWHM so often as it is found a non profile spectrum
                        
                    stdLine = (1/2.3548)*f[0]/nLines     #   estimate the observed standard deviation
                    lineWidthBase = (lambd[lineP['sup'][x]] - lambd[lineP['inf'][x]])/(2*nLines)
                      ##   calculate the observed line width at the base of the profile
                    if ampMax <=0:
                        print "amplitud menor a cero"
                        Gaus.append(models.Gaussian1D(amplitude=0,mean=ListLines['LAMBDA VAC ANG'][x],stddev=0))
                    elif ampMax<=3*spIvar[lineP['lambda'][x]]:
                          ##   no model for the amplitud less tah three times the error
                        Gaus.append(models.Gaussian1D(amplitude=0,mean=ListLines['LAMBDA VAC ANG'][x],stddev=0))
                    else:
                        Gaus.append(models.Gaussian1D(amplitude=ampMax,mean=ListLines['LAMBDA VAC ANG'][x],stddev=stdLine, bounds={'mean':(ListLines['LAMBDA VAC ANG'][x]-lineWidthBase/(2*nLines), ListLines['LAMBDA VAC ANG'][x]+lineWidthBase/(2*nLines)),'stddev':(0.8*stdLine, 1.5*stdLine)}))
                elif ampMax<=3*spIvar[lineP['lambda'][x]]:
                      ##   no model for the amplitud less tah three times the error
                    Gaus.append(models.Gaussian1D(amplitude=0,mean=ListLines['LAMBDA VAC ANG'][x],stddev=0))
        condit = False  #   esto es mejor quitarlo
    sum_Gaussian=Gaus[0]
    #for ooasia in range(1,len(Gaus)):
            #  sum of all the models respective to all the wavelenghts 
    #    sum_Gaussian=sum_Gaussian + Gaus[ooasia]
    sum_Ga2=Gaus[0]+Gaus[1]      #   only two models of the [o II]3726-29 lines
    sum_Gaussian=Gaus[0]+Gaus[1]+Gaus[2]+Gaus[3]+Gaus[4]+Gaus[5]+Gaus[6]+Gaus[7]+Gaus[8]+Gaus[9]+Gaus[10]+Gaus[11]+Gaus[12]+Gaus[13]+Gaus[14]
    gaussian_fit = f1(sum_Gaussian, lambd, ListGal[i]-spec, maxiter=iterr)
    gaussian_fit2 = f1(sum_Ga2, lambd, ListGal[i]-spec, maxiter=iterr) 
    Graph=[]
    Graph.append(gaussian_fit)   
    Graph.append(gaussian_fit2)
    return Graph, lineP, spec     ##  return the fitting to both models and line profiles wavelenghts


# In[2]:

#   ----------------------------------------------------------------------------------------------------------------------
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

#   ----------------------------------------------------------------------------------------------------------------------
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

rc = pn.RedCorr()
rc.law = 'G03 LMC'
IntrinsicHB=np.linspace(2.8,3.1,31)
