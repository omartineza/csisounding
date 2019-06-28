'''
SPHERE_METHODS.py

adjust_longitudes(lon):
distance_sphere(lon1, lat1, lon2, lat2):
true_gcoords_array(rot_longitudes, rot_latitudes, lonp, latp):
true_gcoords(rot_longitudes, rot_latitudes, lonp, latp):

Oscar Martinez-Alvarado
National Centre for Atmospheric Sciences
01/04/2019
'''
import numpy as np

def adjust_longitudes(lon):
    adjustedlon = np.where(lon>180, lon - 360, lon)

    return adjustedlon


def distance_sphere(lon1, lat1, lon2, lat2):
    lambda1 = np.deg2rad(lon1)
    lambda2 = np.deg2rad(lon2)
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    aux = np.cos(phi1) * np.cos(phi2) * np.cos(lambda1 - lambda2) + \
                      np.sin(phi1) * np.sin(phi2)

    aux = np.minimum(aux, 1.)
    aux = np.maximum(aux, -1.)

    s = np.arccos(aux)

    return s

def angle_sphere(lon0, lat0, lon1, lat1, lon2, lat2):
    '''
    Angle on the sphere with vertex on lon0, lat0 and sides ending at 
    lon1, lat1 and lon2, lat2
    '''
    lambda0 = np.deg2rad(lon0)
    lambda1 = np.deg2rad(lon1)
    lambda2 = np.deg2rad(lon2)
    phi0 = np.deg2rad(lat0)
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    r0r1 = np.cos(phi0) * np.cos(lambda0) * np.cos(phi1) * np.cos(lambda1) + \
           np.cos(phi0) * np.sin(lambda0) * np.cos(phi1) * np.sin(lambda1) + \
           np.sin(phi0) * np.sin(phi1)
    r0r2 = np.cos(phi0) * np.cos(lambda0) * np.cos(phi2) * np.cos(lambda2) + \
           np.cos(phi0) * np.sin(lambda0) * np.cos(phi2) * np.sin(lambda2) + \
           np.sin(phi0) * np.sin(phi2)
    r1r2 = np.cos(phi1) * np.cos(lambda1) * np.cos(phi2) * np.cos(lambda2) + \
           np.cos(phi1) * np.sin(lambda1) * np.cos(phi2) * np.sin(lambda2) + \
           np.sin(phi1) * np.sin(phi2)
    t1 = np.sqrt(1.0 - r0r1**2)
    t2 = np.sqrt(1.0 - r0r2**2)
    alpha = np.rad2deg(np.real(np.arccos((r1r2 - r0r1 * r0r2) / (t1 * t2))))

    return alpha


def true_gcoords_array(rot_longitudes, rot_latitudes, lonp, latp):
    lambdap = np.deg2rad(lonp) - np.pi;
    phip = np.pi/2. - np.deg2rad(latp);
    rlambda = np.deg2rad(rot_longitudes)
    rphi = np.deg2rad(rot_latitudes)
    numerador = (np.cos(phip)*np.sin(lambdap)*np.cos(rphi)*np.cos(rlambda) + \
                     np.cos(lambdap)*np.cos(rphi)*np.sin(rlambda) - \
                     np.sin(phip)*np.sin(lambdap)*np.sin(rphi))
    denominador = (np.cos(phip)*np.cos(lambdap)*np.cos(rphi)*np.cos(rlambda) - \
                       np.sin(lambdap)*np.cos(rphi)*np.sin(rlambda) - \
                       np.sin(phip)*np.cos(lambdap)*np.sin(rphi))
    
    true_longitudes = np.rad2deg(np.arctan2(numerador, denominador))
    
    sphi = np.sin(phip)*np.cos(rphi)*np.cos(rlambda) + \
        np.cos(phip)*np.sin(rphi);
    true_latitudes = np.rad2deg(np.arcsin(sphi));
    
    return true_longitudes, true_latitudes

def true_gcoords(rot_longitudes, rot_latitudes, lonp, latp):
    lambdap = np.deg2rad(lonp) - np.pi;
    phip = np.pi/2. - np.deg2rad(latp);
    [rphi, rlambda] = np.meshgrid(rot_latitudes, rot_longitudes);
    rlambda = np.deg2rad(rlambda);
    rphi = np.deg2rad(rphi);
    numerador = (np.cos(phip)*np.sin(lambdap)*np.cos(rphi)*np.cos(rlambda) + \
                     np.cos(lambdap)*np.cos(rphi)*np.sin(rlambda) - \
                     np.sin(phip)*np.sin(lambdap)*np.sin(rphi))
    denominador = (np.cos(phip)*np.cos(lambdap)*np.cos(rphi)*np.cos(rlambda) - \
                       np.sin(lambdap)*np.cos(rphi)*np.sin(rlambda) - \
                       np.sin(phip)*np.cos(lambdap)*np.sin(rphi))
    
    true_longitudes = np.rad2deg(np.arctan2(numerador, denominador))
    
    sphi = np.sin(phip)*np.cos(rphi)*np.cos(rlambda) + \
        np.cos(phip)*np.sin(rphi);
    true_latitudes = np.rad2deg(np.arcsin(sphi));
    
    return true_longitudes, true_latitudes

