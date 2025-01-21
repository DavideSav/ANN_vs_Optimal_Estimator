# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:19:50 2023

@author: Daniele
"""


def load_atm__prf(fn):
    import netCDF4 as nc
    from svp import svp
    import numpy as np
    #fn='afgl_1986-midlatitude_summer.nc'
    prf=nc.Dataset(fn)
    t=np.asarray(prf['t'][:].filled())
    z=np.asarray(prf['z'][:].filled())
    p=np.asarray(prf['p'][:].filled())
    vol_frac_h2o=np.asarray(prf['x_H2O'][:].filled())
    partial_pressure_h20=vol_frac_h2o*p
    saturation_vapor_pressure=svp(t)
    relative_humidity=partial_pressure_h20/saturation_vapor_pressure*100
    
   
    
    
    return z,p,t,relative_humidity
    
    
    