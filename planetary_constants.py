import iris

a = 6371229.0 # m, Earth's radius
Omega = 7.292e-5 # rad s-1, Earth's angular speed of rotation
R = 287. # J K-1 kg-1, Gas constant dry air
R_V = 461.5 # J K-1 kg-1, Gas constant vapour
CP = 1004. # J K-1 kg-1, Specific heat capacity at constant pressure
CV = 717. # J K-1 kg-1
LV = 2.5e6 # J kg-1
PZERO = 1000. # hPa
TZERO = 273.15 # K
GRAVITY = 9.80665 # m s-2
M_D = 0.02896 # kg mol-1, Mean molar weight dry air

# Moist (water vapour) constants
M_V = 0.018015 # kg mol-1, Mean molar weight water

def earth_radius():
   a0 = iris.coords.AuxCoord(a, long_name='Earth_radius', units='m')

   return a0

def reference_pressure():
   p0 = iris.coords.AuxCoord(PZERO, long_name='reference_pressure', \
                                 units='hPa') 
   return p0

def air_gas_constant():
   r = iris.coords.AuxCoord(R, long_name='air_gas_constant', \
                            units='J K-1 kg-1')
   return r

def KAPPA():
   K = R/CP
   return K

def EPSILON():
   E = M_V / M_D
   return E

