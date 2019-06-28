import numpy as np

import cf_units

import iris as iris

def gradient_sphere(field):
    '''
    GRADIENT_SPHERE Gradient on spherical coordinates.
      [FX,FY,MAGF] = GRADIENT_SPHERE(LON,LAT,FIELD,JLAT,KLEV) returns the 
      components of the gradient in spherical coordinates, FX, FY, and the
      magnitude of the gradient MAGF = sqrt(FX.^2 + FY.^2). 
      The result is in field_units per km

    Oscar Martinez-Alvarado
    NCAS-Atmospheric Physics and
    Department of Meteorology
    University of Reading

    Created on 12/Dec/2016
    '''

    from planetary_constants import earth_radius

    lon = field.coord('longitude')
    lat = field.coord('longitude')

    dx = np.deg2rad(lon.points[1] - lon.points[0])
    dy = np.deg2rad(lat.points[1] - lat.points[0])

    Fy, Fx = np.gradient(field.data, dy, dx, axis=(1, 2))

    Klev, Jlat, Ilon = field.data.shape

    for jj in range(Jlat):
        Fx[:, jj, :] = Fx[:, jj, :] / np.cos(np.deg2rad(lat.points[jj]))

    a = earth_radius()
    a.convert_units('km')

    gradx = field.copy(Fx)
    gradx = gradx/a
    gradx.rename('gradient_x-component')

    grady = field.copy(Fy)
    grady = grady/a
    grady.rename('gradient_y-component')

    magF = (gradx**2 + grady**2)**0.5
    magF.rename('gradient_magnitude')
    
    return gradx, grady, magF

def d_dLp(field):
    field_data = field.data

    pressure = field.coord('pressure').points

    Lp = np.log(pressure)

    Klev, Jlat, Ilon = field_data.shape
    df_dLp = np.zeros_like(field_data)
    for jj in range(Jlat):
        for ii in range(Ilon):
            df_dLp[1:-1, jj, ii] = (field_data[2:, jj , ii] - \
                                    field_data[:-2, jj, ii]) / \
                (Lp[2:] - Lp[:-2])
            df_dLp[0, jj, ii] = (field_data[1, jj ,ii] - \
                                 field_data[0, jj, ii]) / (Lp[1] - Lp[0])
            df_dLp[-1, jj, ii] = (field_data[-1, jj, ii] - \
                                  field_data[-2, jj, ii]) / (Lp[-1] - Lp[-2])

    ofield = field.copy(df_dLp)
    ofield.rename('vertical derivative') 
    
    return ofield

def p3d_like(theta):
    # Creates an auxiliary cube with the same shape of theta but containing 
    # pressure data (constant on a pressure level)
    pressure = theta.coord('pressure')
    Ntsteps, Klev, Jlat, Ilon = theta.data.shape

    # Auxiliary 3d (4d) pressure cube
    p3d_data = np.zeros_like(theta.data)
    for kk in range(Klev):
        p3d_data[:, kk, :, :] = pressure.points[kk]

    p3d = theta.copy(p3d_data)
    p3d.rename('air_pressure')
    p3d.units = pressure.units

    return p3d

def ezwbpottemp(pressure, qs, theta):
    '''
    WBPOTTEMP Wet-bulb potential temperature.
      THW = WBPOTTEMP(P, QS, T) returns the wet-bulb potential temperature
      given pressure P (in hPa), mixing ratio QS (in kg/kg) and
      temperature T (in K). P can be either an array with of the same size as
      T and QS or a scalar.

    Oscar Martinez-Alvarado
    Department of Meteorology
    University of Reading

    Created on 04/Dec/2008
    Modified 12/Dec/2016: O.M-A. 
    The definition of constants KAPPA, PZERO, TZERO is taken from 
    atmos_constants.m
    '''
    from planetary_constants import TZERO

    def dQdx(X):
        Y = 0.622 * 6.112 * np.exp(17.67*X / (X + 243.5)) * \
        (17.67 / (X+243.5) - 17.67 * X /(X + 243.5)**2)
        
        return Y

    temp = temperature(pressure, theta)
    temp.convert_units('kelvin')

    const = np.log(theta.data) + 2491.0*qs.data/temp.data
    T0 = temp.data
    delta = 1.0
    while delta > 1e-5:
        Q0 = ezqsat_data(1000.0, T0)
        T1 = T0 * (1.0 - \
            (T0*(np.log(T0) + \
            2491.0*Q0/T0 - const))/ \
            (T0 * (1.0 + 2491.0 * dQdx(T0-TZERO)) - 2491.0 * Q0))
        ERR = np.absolute(T1 - T0)
        delta = ERR.max()
        T0 = T1

    thw = theta.copy(T1)
    thw.rename('wet_bulb_potential_temperature')
    thw.units = cf_units.Unit('kelvin')

    return thw

def temperature(pressure, theta):
    from planetary_constants import reference_pressure, KAPPA

    p0 = reference_pressure()
    p0.convert_units(pressure.units)
    exner = (pressure / p0)**KAPPA()
    exner.rename('dimensionless_exner_function')
    
    temp = exner * theta
    temp.rename('air_temperature')

    return temp

def mixing_ratio(qv):
    q_data = qv.data/(1.0 - qv.data)
    q = qv.copy(q_data)
    q.rename('mixing_ratio')
    q.units = cf_units.Unit('kg kg-1')

    return q

def esat_purewater(temperature):
    temperature.convert_units('celsius')
    log10ew_data = (0.7859 + 0.03477 * temperature.data) / \
                   ( 1.0 + 0.00412 * temperature.data)
    ew_data = 10.0**(log10ew_data)
    ew = temperature.copy(ew_data)
    ew.rename('saturation_vapour_pressure_pure_water')
    ew.units = cf_units.Unit('hPa')

    return ew

def saturation_mixing_ratio_water(pressure, temperature):
    from planetary_constants import EPSILON

    temperature.convert_units('celsius')
    Fw = 1.0 + 1.0e-6 * pressure * (4.5 + 6.0e-4 * temperature**2)
    esw_purewater = esat_purewater(temperature)
    esice = Fw * esw_purewater
    esice.rename('saturation_pressure_over_ice')
    qsat_data = EPSILON() * esice.data / \
                (np.maximum(pressure.data, esice.data) - \
                 (1.0 - EPSILON())*esice.data)
    qsat = temperature.copy(qsat_data)
    qsat.rename('saturation_mixing_ratio')
    qsat.units = cf_units.Unit('kg kg-1')

    return qsat

def saturation_mixing_ratio_ice(pressure, temperature):
    from planetary_constants import EPSILON
    import scipy.io as sio

    # Saturation vapour pressure look-up table, given in Pa
    indat = sio.loadmat('eswrtice')
    ES = indat['ES'].flatten()

    # Reference temperature array corresponding to ES
    T_LOW = 183.15
    T_HIGH = 338.15
    DELTA_T = 0.1

    T = np.arange(T_LOW, T_HIGH+DELTA_T/2.0, DELTA_T)

    temperature.convert_units('celsius')
    Fw = 1.0 + 1.0e-6 * pressure * (4.5 + 6.0e-4 * temperature**2)
    Fw.units = '1'

    temperature.convert_units('kelvin')
    esice_purewater_data = np.interp(temperature.data, T, ES[1:-1])
    esice_purewater = temperature.copy(esice_purewater_data)
    esice_purewater.rename('saturation_vapour_pressure_pure_water')
    esice_purewater.units = cf_units.Unit('Pa')

    esice = Fw * esice_purewater
    esice.rename('saturation_pressure_over_ice')
    esice.convert_units('hPa')
    qsat_data = EPSILON() * esice.data / \
                (np.maximum(pressure.data, esice.data) - \
                 (1.0 - EPSILON())*esice.data)
    qsat = temperature.copy(qsat_data)
    qsat.rename('saturation_mixing_ratio')
    qsat.units = cf_units.Unit('kg kg-1')

    return qsat

def rhwrtwater(pressure, temperature, qv):
    '''
    Based on RHWRTWATER_TEMP.m (see ~sws07om/matlabd/functions):
    RHWRTWATER returns relative humidity with respect to water. TEMPERATURE
    is given in Kelvin, P in hPa, and QV in kg kg-1.
    '''
    # Mixing ratio
    q = mixing_ratio(qv)

    # Saturation mixing ratio
    qsat = saturation_mixing_ratio_water(pressure, temperature)
    rh = 100.0 * q / qsat
    rh.rename('relative_humidity_with_respect_to_ice')

    rh = rh.copy(np.where(rh.data<0.0, 0.0, rh.data))

    return rh

def rhwrtice(pressure, temperature, qv):
    '''
    Based on RHWRTICE_TEMP_PARALLEL.m (see ~sws07om/matlabd/functions):
    RHWRTICE returns relative humidity with respect to water if TEMPERATURE
    is above freezing temperature TZERO and with respect to ice if
    TEMPERATURE is below TZERO. TEMPERATURE is given in Kelvin, P in hPa, and
    QV in kg kg-1.

    Oscar Martinez-Alvarado
    Department of Meteorology
    University of Reading

    Modified 12/Dec/2016: O.M-A. 
    The definition of constants EPSILON and TZERO is taken from 
    atmos_constants.m
    '''
    # Mixing ratio
    q = mixing_ratio(qv)

    # Saturation mixing ratio
    qsat = saturation_mixing_ratio_ice(pressure, temperature)

    rh = 100.0 * q / qsat
    rh.rename('relative_humidity_with_respect_to_ice')

    return rh

def ezqsat(pressure, temperature):
    '''
    Based on EZQSAT_PARALLEL.m (see ~sws07om/matlabd/functions):
    Temperature on input should be in Kelvin.
    Pressure should be in hPa.

    Oscar Martinez-Alvarado
    Department of Meteorology
    University of Reading

    Modified 12/Dec/2016: O.M-A. 
    The definition of constants EPSILON and TZERO is taken from 
    atmos_constants.m
    '''
    from planetary_constants import EPSILON

    temperature.convert_units('celsius')
    es = 6.112 * \
         iris.analysis.maths.exp(17.67 * temperature / (temperature + 243.5))
    # Saturation mixing ratio
    qsat_data = EPSILON() * es.data / \
                (np.maximum(pressure.data, es.data) - \
                 (1.0 - EPSILON())*es.data)
    qsat = temperature.copy(qsat_data)
    qsat.rename('saturation_mixing_ratio')
    qsat.units = 'kg kg-1'
    
    return qsat
    
def esat_purewater_data(temperature):
    ''''
    Temperature on input should be in Kelvin.
    Returns saturation vapour pressure, in hPa.
    '''
    from planetary_constants import EPSILON, TZERO

    temperature = temperature - TZERO
    log10ew = (0.7859 + 0.03477 * temperature) / \
                   ( 1.0 + 0.00412 * temperature)
    ew = 10.0**(log10ew)

    return ew

def saturation_mixing_ratio_water_data(pressure, temperature):
    ''''
    Temperature on input should be in Kelvin.
    Pressure should be in hPa.
    '''
    from planetary_constants import EPSILON, TZERO

    tC =  temperature - TZERO
    Fw = 1.0 + 1.0e-6 * pressure * (4.5 + 6.0e-4 * tC**2)
    esw_purewater = esat_purewater_data(temperature)
    esice = Fw * esw_purewater
    qsat = EPSILON() * esice / \
                (np.maximum(pressure, esice) - (1.0 - EPSILON())*esice)

    return qsat

def ezqsat_data(pressure, temperature):
    '''
    Based on EZQSAT_PARALLEL.m (see ~sws07om/matlabd/functions):
    Temperature on input should be in Kelvin.
    Pressure should be in hPa.

    Oscar Martinez-Alvarado
    Department of Meteorology
    University of Reading

    Modified 12/Dec/2016: O.M-A. 
    The definition of constants EPSILON and TZERO is taken from 
    atmos_constants.m
    '''
    from planetary_constants import EPSILON, TZERO

    tempC = temperature - TZERO
    es = 6.112 * np.exp(17.67 * tempC / (tempC + 243.5))
    qsat_data = EPSILON() * es / \
                (np.maximum(pressure, es) - (1.0 - EPSILON())*es)
    
    return qsat_data

def dcape_sounding(p, t, q, qs=None):
    '''
    Based on the FORTRAN program calcsound.f by Kerry Emanuel available from
    https://emanuel.mit.edu/fortran-programs-atmospheric-convection
    Temperature on input should be in Kelvin
    '''

    from planetary_constants import R, R_V, CP, TZERO, EPSILON

    # Other constants
    CPVMCL = 2320.0
    CL = 4190.0
    ALV0 = 2.501e6

    start_lev = 0
    tolerance = 1e-5

    nlevels = p.shape[0]

    if qs is None:
        qs = saturation_mixing_ratio_water_data(p, t)

    # Define pseudo-adiabatic entropy SP
    alvn = ALV0 - CPVMCL*(t[start_lev] - TZERO)
    ah = (CP + q[start_lev]*CL)*t[start_lev] + alvn*q[start_lev]
    
    #   ***  Find the temperature and mixing ratio of the parcel at   ***
    #   ***    level I saturated by a wet bulb process                ***

    slope = CP + qs[start_lev]*alvn**2 / (R_V*t[start_lev]**2)
    tg = t[start_lev]
    qg = qs[start_lev]

    iterate = True
    while iterate:
        alv1 = ALV0 - CPVMCL*(tg-TZERO)
        ahg = (CP + CL*qg)*tg + alv1*qg
        tg_correction = (ah - ahg)/slope
        tg = tg + tg_correction
        tc = tg - TZERO
        enew = 6.112*np.exp(17.67*tc/(243.5 + tc))
        qg = EPSILON()*enew/(p[start_lev]-enew)
        if np.abs(tg_correction) < tolerance:
            iterate = False;
    
    # Calculate conserved variable at top of downdraught
    eg = qg*p[start_lev]/(EPSILON() + qg)
    Spd = CP*np.log(tg) - R*np.log(p[start_lev]-eg) + alv1*qg/tg
    tvd = tg*(1.0 + qg/EPSILON())/(1.0 + qg) - \
          t[start_lev]*(1 + q[start_lev]/EPSILON())/(1.0 + q[start_lev]);
    if p[start_lev] < 100:
        tvd = 0

    qgd0 = qg
    tgd0 = tg

    # Begin downdraught loop
    suma = 0
    qp = np.zeros((nlevels, ));
    tlp = np.zeros((nlevels, ));
    # Calculate estimates of the rates of change of the entropies with
    # temperature at constant pressure
    alv = ALV0 - CPVMCL*(t - TZERO);
    slp = (CP + qs*CL + qs*alv**2/(R_V*t**2))/t;
    tv = t*(1 + q/EPSILON())/(1 + q)
    #  Begin downdraught loop   ***
    for kk in np.arange(start_lev+1, nlevels):
        # Iteratively calculate lifted parcel temperature and mixing ratios
        # for pseudo-adiabatic descent
        tg=t[kk]
        qg=qs[kk]
        iterate = True
        no_iterations = 0
        while iterate and no_iterations <= 100:
            cpw = suma + 0.5*CL*(qgd0 + qg)*(np.log(tg) - np.log(tgd0))
            em = qg*p[kk]/(EPSILON() + qg);
            alvg = ALV0 - CPVMCL*(tg-TZERO);
            spg = CP*np.log(tg) - R*np.log(p[kk]-em) + cpw + alvg*qg/tg;
            tg_correction = (Spd - spg)/slp[kk];
            tg = tg + tg_correction;
            tc = tg - TZERO;
            enew = 6.112*np.exp(17.67*tc/(243.5 + tc));
            qg = EPSILON()*enew/(p[kk] - enew)
            if np.abs(tg_correction) <= tolerance: 
                iterate = False
            
            no_iterations = no_iterations + 1

        if np.abs(tg_correction) > tolerance:
            print('WARNING: Failed convergence: %d, %5.1e' % \
                  (kk, np.abs(tg_correction) - tolerance))
        
        qgd0 = qg
        tgd0 = tg
        suma = cpw
        tlp[kk] = tg
        qp[kk] = qg
    
    tlvp = tlp*(1 + qp/EPSILON())/(1.0 + qp)
    tvpdif = tlvp - tv
    if p[start_lev]<100:
        tvpdif = 0
    
    tvpdif = np.minimum(tvpdif, 0.0)
    tvdifm = np.insert(tvpdif[start_lev+1:-1], 0, tvd)
    tvm = (tvpdif[1:] + tvdifm)/2.0
    pm = (p[:-1] + p[1:])/2.0
    delta_p = p[start_lev+1:] - p[start_lev:-1]
    dcape = -R * np.sum( \
                         tvm[np.where(tvm<0)] * delta_p[np.where(tvm<0)] / \
                         pm[np.where(tvm<0)])
   
    return dcape


