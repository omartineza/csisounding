import numpy as np

import cf_units

import iris as iris

def d_dX(Xcart, Ycart, u, v, field, wspeed=None):
    # Partial derivative with respect to X, where X is the distance orthogonal 
    # to the shear vector
    if wspeed is None:
        wspeed = (u**2 + v**2)**0.5
        wspeed.rename('wind_speed')
    
    field_d = field.data

    Klev, Jlat, Ilon = field_d.shape
    df_dXcart = np.zeros_like(field_d)
    df_dYcart = np.zeros_like(field_d)
    eta_data = np.zeros_like(field_d)
    for kk in range(Klev):
        # Compute partial wrt Xcart
        df_dXcart[kk, :, 1:-1] = (field_d[kk, :, 2:] - field_d[kk, :, :-2]) / \
                             (Xcart[:, 2:] - Xcart[:, :-2])
        df_dXcart[kk, :, 0] = (field_d[kk, :, 1] - field_d[kk, :, 0]) / \
                          (Xcart[:, 1] - Xcart[:, 0])
        df_dXcart[kk, :, -1] = (field_d[kk, :, -1] - field_d[kk, :, -2]) / \
                           (Xcart[:, -1] - Xcart[:, -2])
        # Compute partial wrt Ycart
        df_dYcart[kk, 1:-1, :] = (field_d[kk, 2:, :] - field_d[kk, :-2, :]) / \
                             (Ycart[2:, :] - Ycart[:-2, :])
        df_dYcart[kk, 0, :] = (field_d[kk, 1, :] - field_d[kk, 0, :]) / \
                          (Ycart[1, :] - Ycart[0, :])
        df_dYcart[kk, -1, :] = (field_d[kk, -1, :] - field_d[kk, -2, :]) / \
                           (Ycart[-1, :] - Ycart[-2, :])

    df_dX_data = (v.data*df_dXcart - u.data*df_dYcart) / wspeed.data

    df_dX = u.copy(df_dX_data)
    df_dX.rename('horizontal_derivative')
    df_dX.units = field.units * cf_units.Unit('m-1')

    return df_dX
    
def vorticity(X, Y, u, v, Coriolis, wspeed=None):
    if wspeed is None:
        wspeed = (u**2 + v**2)**0.5
        wspeed.rename('wind_speed')

    Lwspeed = np.log(wspeed.data)

    Klev, Jlat, Ilon = Lwspeed.shape
    dLws_dX = np.zeros_like(Lwspeed)
    dLws_dY = np.zeros_like(Lwspeed)
    eta_data = np.zeros_like(Lwspeed)
    for kk in range(Klev):
        # Compute partial wrt X
        dLws_dX[kk, :, 1:-1] = (Lwspeed[kk, :, 2:] - Lwspeed[kk, :, :-2]) / \
                     (X[:, 2:] - X[:, :-2])
        dLws_dX[kk, :, 0] = (Lwspeed[kk, :, 1] - Lwspeed[kk, :, 0]) / \
                     (X[:, 1] - X[:, 0])
        dLws_dX[kk, :, -1] = (Lwspeed[kk, :, -1] - Lwspeed[kk, :, -2]) / \
                     (X[:, -1] - X[:, -2])
        # Compute partial wrt Y
        dLws_dY[kk, 1:-1, :] = (Lwspeed[kk, 2:, :] - Lwspeed[kk, :-2, :]) / \
                     (Y[2:, :] - Y[:-2, :])
        dLws_dY[kk, 0, :] = (Lwspeed[kk, 1, :] - Lwspeed[kk, 0, :]) / \
                     (Y[1, :] - Y[0, :])
        dLws_dY[kk, -1, :] = (Lwspeed[kk, -1, :] - Lwspeed[kk, -2, :]) / \
                     (Y[-1, :] - Y[-2, :])
    
    eta_data = v.data*dLws_dX - u.data*dLws_dY + Coriolis

    eta = u.copy(eta_data)
    eta.rename('vorticity')
    eta.units = cf_units.Unit('s-1')

    return eta

