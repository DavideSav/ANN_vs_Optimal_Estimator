

def gas_abs(pressurePa,tempK,relative_hum,frequency):
    import numpy as np
 

#%%%%%   INPUTS %%%%
#% pressurePa in Pa
#% tempK temperature in K
#% frequency in GHz
#% relative humidity in %
#% range in m
#
#%%%%  OUTPUTS %%%
#% attenuation coefficients  in Np/m 
#%

#     Rosenkranz, P. W. (1998). Water vapor microwave continuum absorption: 
#    A comparison of measurements and models. Radio Science, 33(4), 919-928.
 
#%%% constants  used %%%%
    c = 299792458  # speed of light
    pi = 3.141592653589793 
    t_abs = 273.15
    g = 9.80665
    sky_temp = 2.73
    rho_water = 1000

    tpt = 273.16 # triple point temperature
    estpt = 611.14 # saturation vapor triple point temperature
    mmd = 28.9644e-3  # molar mass of dry air
    mmv = 18.0153e-3  # molar mass of vapor
    k_b = 1.380658e-23 # Boltzman constant
    n_a = 6.0221367e+23 # Avogadro number
    r_d = 287.0596736665907 #  gas constant of dry air
    r_v = 461.5249933083879 #  gas constant of water vapor
    vapor_hc  = 2.5008e+6   #  vaporization heat constant
    sublim_hc  = 2.8345e+6  #  sublimation heat constant
    water=0                 # liquid water not taken into account
 



#% % K ! tempK in K
#    temperature in °C
#    tempK=temperature+t_abs
    temperature=tempK-t_abs
    
# pressurePa in Pascal
# pressure in mbar
    pressure = pressurePa/100    
    es        = 6.112*np.exp((17.67*temperature)/(temperature+243.5))  #  saturation vapor pressure
    e         = (relative_hum*es)/100   # vapor pressure
    w         = (.622)*e/(pressure-e)
    rho_vap   = w*((pressure*1e+2))/(278*tempK)
 
# NITROGEN

    th = 300/tempK
    absn2 = 6.4e-14*pressure*pressure*frequency*frequency*th**3.55

# Parameters for o2 and h2o vapor attenuation calculation

    nbands_o2 = 40
    nbands_h2o = 15
#%%%%%%%%%%%%%%%   

    x1 = 0.8
    wb300 = 0.56
  
    w300 = [1.63, 1.646, 1.468, 1.449, 1.382, 1.360,1.319, 1.297, 1.266, 1.248, 1.221, 1.207, 1.181, 1.171,1.144, 1.139, 1.110, 1.108, 1.079, 1.078, 1.05, 1.05,1.02, 1.02, 1.0, 1.0, 0.97, 0.97, 0.94, 0.94, 0.92, 0.92,0.89, 0.89, 1.92, 1.92, 1.92, 1.81, 1.81, 1.81]
    y300 = [-0.0233,  0.2408, -0.3486,  0.5227,-0.5430,  0.5877, -0.3970,  0.3237, -0.1348,  0.0311,0.0725, -0.1663,  0.2832, -0.3629, 0.3970, -0.4599,0.4695, -0.5199,  0.5187, -0.5597,0.5903,-0.6246,0.6656, -0.6942,  0.7086, -0.7325,  0.7348, -0.7546,0.7702, -0.7864, 0.8083, -0.8210,  0.8439, -0.8529,0., 0., 0., 0., 0., 0.]
    v =[0.0079, -0.0978,  0.0844, -0.1273, 0.0699, -0.0776,  0.2309, -0.2825,0.0436,-0.0584,0.6056,-0.6619,0.6451, -0.6759,  0.6547, -0.6675,0.6135, -0.6139,  0.2952, -0.2895,0.2654,-0.2590,0.3750, -0.3680,  0.5085, -0.5002,  0.6206, -0.6091,0.6526, -0.6393,  0.6640, -0.6475,  0.6729, -0.6545,0., 0., 0., 0., 0., 0]
       
    #%      LINES ARE ARRANGED 1-,1+,3-,3+,ETC. IN SPIN-ROTATION SPECTRUM
    f    = [118.7503, 56.2648, 62.4863, 58.4466, 60.3061, 59.5910,59.1642, 60.4348, 58.3239, 61.1506, 57.6125, 61.8002,56.9682,62.4112, 56.3634, 62.9980, 55.7838, 63.5685,55.2214, 64.1278, 54.6712, 64.6789, 54.1300, 65.2241,53.5957,65.7648, 53.0669, 66.3021, 52.5424, 66.8368,52.0214, 67.3696, 51.5034, 67.9009, 368.4984, 424.7631,487.2494, 715.3932, 773.8397, 834.1453]
    s300 = [0.2936E-14, 0.8079E-15, 0.2480E-14, 0.2228E-14,0.3351E-14, 0.3292E-14, 0.3721E-14, 0.3891E-14,0.3640E-14, 0.4005E-14, 0.3227E-14, 0.3715E-14,0.2627E-14, 0.3156E-14, 0.1982E-14, 0.2477E-14,0.1391E-14, 0.1808E-14, 0.9124E-15, 0.1230E-14,0.5603E-15, 0.7842E-15, 0.3228E-15, 0.4689E-15,0.1748E-15, 0.2632E-15, 0.8898E-16, 0.1389E-15,0.4264E-16, 0.6899E-16, 0.1924E-16, 0.3229E-16,0.8191E-17, 0.1423E-16, 0.6460E-15, 0.7047E-14,0.3011E-14, 0.1826E-14, 0.1152E-13, 0.3971E-14]
    be   = [0.009, 0.015, 0.083, 0.084, 0.212, 0.212, 0.391, 0.391, 0.626, 0.626,0.915, 0.915, 1.26, 1.26, 1.66, 1.665, 2.119, 2.115, 2.624, 2.625,3.194, 3.194, 3.814, 3.814, 4.484, 4.484, 5.224, 5.224,6.004, 6.004,6.844, 6.844,7.744, 7.744, 0.048, 0.044, 0.049, 0.145, 0.141, 0.145]

     
    th1     = th-1
    b       = th**x1
    preswv  = rho_vap*1e+3*tempK/217.
    presda  = pressure-preswv
    den     = 0.001*(presda*b + 1.1*preswv*th)
    dfnr    = wb300*den
    loc_sum = 1.6e-17*frequency*frequency*dfnr/(th*(frequency*frequency + dfnr*dfnr))
 
    for k in range(nbands_o2):
            df = w300[k]*den
            y = 0.001*pressure*b*(y300[k]+v[k]*th1)
            str0 = s300[k]*np.exp(-be[k]*th1)
            sf1 = (df + (frequency-f[k])*y)/((frequency-f[k])**2 + df*df)
            sf2 = (df - (frequency+f[k])*y)/((frequency+f[k])**2 + df*df)
            loc_sum = loc_sum + str0*(sf1+sf2)*(frequency/f[k])**2
       
       
    abso2 = 0.5034e12*loc_sum*presda*th**3/pi
#%%%%%%%% COMPUTE ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR
        
# %     LINE FREQUENCIES:
    fl = [22.2351, 183.3101, 321.2256, 325.1529, 380.1974, 439.1508,443.0183, 448.0011, 470.8890, 474.6891, 488.4911, 556.9360,620.7008, 752.0332, 916.1712]
      #  %     LINE INTENSITIES AT 300K:
    s1 = [0.1310E-13, 0.2273E-11, 0.8036E-13, 0.2694E-11, 0.2438E-10, 0.2179E-11, 0.4624E-12, 0.2562E-10, 0.8369E-12, 0.3263E-11,0.6659E-12, 0.1531E-08, 0.1707E-10, 0.1011E-08, 0.4227E-10]
      #  %     T COEFF. OF INTENSITIES:
    b2 = [2.144, .668, 6.179, 1.541, 1.048, 3.595, 5.048, 1.405,3.597, 2.379, 2.852, .159, 2.391, .396, 1.441]
      #  %     AIR-BROADENED WIDTH PARAMETERS AT 300K:
    w3 = [0.002656, 0.00281, 0.0023, 0.00278, 0.00287, 0.0021, 0.00186, 0.00263, 0.00215, 0.00236, 0.0026, 0.00321, 0.00244, 0.00306, 0.00267]
      #  %     T-EXPONENT OF AIR-BROADENING:
    x = [0.69, 0.64, 0.67, 0.68, 0.54, 0.63, 0.60, 0.66, 0.66, 0.65, 0.69, 0.69, 0.71, 0.68, 0.70]
      #  %     SELF-BROADENED WIDTH PARAMETERS AT 300K:
    ws = [0.0127488, 0.01491, 0.0108, 0.0135, 0.01541, 0.0090, 0.00788,0.01275, 0.00983, 0.01095, 0.01313, 0.01320, 0.01140, 0.01253, 0.01275]
      #  %     T-EXPONENT OF SELF-BROADENING:
    xs = [0.61, 0.85, 0.54, 0.74, 0.89, 0.52, 0.50, 0.67, 0.65, 0.64, 0.72,1.0, 0.68, 0.84, 0.78]

    if np.any(rho_vap <= 0.):
            abs_rose_98 =0
            return abs_rose_98
    #        return
    
        
        
    pvap = rho_vap*tempK*1000/217. 
    pda = pressure - pvap
    den = 3.335e16 * rho_vap*1000.
    ti = 300/ tempK
    ti2 = ti**2.5
        #%
        #%      continuum terms
    con = (5.43e-10*1.105*pda*ti**3 + 1.8e-8*0.79*pvap*ti**7.5)*pvap*frequency*frequency
        #%
        #%      add resonances
    loc_sum1 = 0.
    df1=np.zeros((2,))
    for ih2o in range(nbands_h2o):
            width = w3[ih2o]*pda*ti**x[ih2o] + ws[ih2o]*pvap*ti**xs[ih2o]
            wsq = width*width
            s = s1[ih2o]*ti2*np.exp(b2[ih2o]*(1.-ti))
            df1[0] = frequency - fl[ih2o]
            df1[1] = frequency + fl[ih2o]
            #%  use clough's definition of local line contribution
            base = width/(562500 + wsq)
            #%  do for positive and negative resonances
            res = 0.
            for j in range(2):
                if(abs(df1[j])<750.) :
                    res = res + width/(df1[j]**2+wsq) - base
                
            
            loc_sum1 = loc_sum1 + s*res*(frequency/fl[ih2o])**2
        
    absh2o = 0.3183e-4*den*loc_sum1 + con
 

   
#        %     COMPUTES ABSORPTION IN NEPERS/KM BY SUSPENDED WATER DROPLETS
#        %     FROM EQUATIONS OF LIEBE, HUFFORD AND MANABE
#        %     (INT. J. IR & MM WAVES V.12(17) JULY 1991
#        %     water IN G/M**3
#        %     frequency  IN GHZ     (VALID FROM 0 TO 1000 GHZ)
#        %     tempK IN KELVIN
#        %     PWR 8/3/92
#        %

         

    if(water <= 0.)  :
            abliq = 0. 
    else :
       
        theta1 = 1.-300/tempK
        eps0 = 77.66 - 103.3*theta1
        eps1 = 0.0671*eps0
        eps2 = 3.52 + 7.52*theta1
        fp = (316*theta1 + 146.4)*theta1 +20.20
        fs = 39.8*fp
        eps = (eps0-eps1)/ (  1+1j*frequency/fp) +(eps1-eps2)/(1+1j*frequency/fs) +eps2
        re = (eps-1)/(eps+2)
        abliq = -0.06286*np.imag(re)*frequency*water

        


        #% in 1/km
    abs_rose_98 =(absn2+abso2 + absh2o)
    # conversion to Np/m
    abs_rose_98 =(absn2+abso2 + absh2o)/1.0e3;  
    return abs_rose_98

#%  
#%  
#%  
#% 
#%  
 
    

#%  
#%  % From ground up
#% for e=1:size(temperature,1)
#% 
#%    tot_tau=abs_rose_98(e,1)*range(1);
#%    att_rose98(e,1)=tot_tau;
#% 
#%        for r=2:size(temperature,2);
#% 
#%              att_rose98(e,r)  = tot_tau+(abs_rose_98(e,r)*nanmean(diff(range)));
#%              tot_tau=att_rose98(e,r);
#% 
#%         end
#% 
#% end
# 
#end
      
