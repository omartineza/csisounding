import pandas as pd
import datetime as dt

work='/home/users/omartineza/sjwork/'
trackdir='/home/users/omartineza/trackdir/'
hemisphere='NH'

runids={ 'ajthm': 'present climate - 2D fields only', \
             'ajthp': 'present climate', \
             'ajthr': 'future climate time slice' }

seasons={ 'mam': {'long_name': 'MAM', \
                      'multiple_file': False}, \
              'jja': {'long_name': 'JJA', \
                          'multiple_file': False}, \
              'son': {'long_name': 'SON', \
                          'multiple_file': False}, \
              'djf': {'long_name': 'DJF', \
                          'multiple_file': False}, \
              's-m': {'long_name': 'September to May', \
                          'multiple_file': True, \
                          'partial_season': ['son', 'djf', 'mam'], \
                          'year_offset': [-1, 0, 0]} }

regions={ 'nh': {'name': 'Northern Hemisphere', \
                     'def': [-180, 180, 0, 90], \
                     'maxrv': [-180, 180, 35, 70], \
                     'start': [-180, 180, 20, 70]}, \
              'natl': {'name': 'North Atlantic', \
                           'def': [-130, 60, 0, 90], \
                           'maxrv': [-80, 40, 45, 70], \
                           'start': [-130, 10, 20, 70]}}

def geolocatetracks(TRACKS, tracking_var, lonrange=[0, 360]):
    import numpy as np

    Zero = 0
    OneEighty = 180
    ThreeSixty = 360

    if lonrange == [0, 360] or lonrange == [-180, 180]:
        track_num = len(TRACKS)
        minlon = 180
        maxlon = -180
        minlat = 90
        maxlat = 0
        for ii in range(track_num):
            lons = TRACKS[ii][tracking_var].lon.values
            lats = TRACKS[ii][tracking_var].lat.values
            if lonrange == [-180, 180] and np.any(lons > OneEighty):
                lons = np.where(lons <= OneEighty, lons, lons - ThreeSixty)
            elif lonrange == [0, 360] and np.any(lons < Zero):
                lons = np.where(lons > Zero, lons, ThreeSixty - lons)
            if lons.max() > maxlon:
                maxlon = lons.max()
            if lons.min() < minlon:
                minlon = lons.min()
            if lats.max() > maxlat:
                maxlat = lats.max()
            if lats.min() < minlat:
                minlat = lats.min()

        return minlon, maxlon, minlat, maxlat
    else:
        print 'ERROR: The only two valid longitude ranges are [-180, 180] or [0, 360]'
        return np.nan, np.nan, np.nan, np.nan
    
def writetracks(TRACKS, filename, \
                fieldnames = ['fvort', 'wind925', 'mslp', 'wind10m']):
    track_num = len(TRACKS)
    add_fld = len(fieldnames)
    with open(filename,'w') as outfile:
        outfile.write('0\n0 0\nTRACK_NUM %d ADD_FLD %d 12 &1111\n' % \
                          (track_num, add_fld))
        for ii in range(track_num):
            outfile.write('TRACK_ID %d START_TIME %s\nPOINT_NUM %d\n' % \
                              (ii+1, \
                                   TRACKS[ii]['vort'].index.tolist()[0], \
                                   len(TRACKS[ii]['vort'].index)))
            xTRACKS = TRACKS[ii]['vort']
            for ff in range(add_fld):
                FLD = TRACKS[ii][fieldnames[ff]][['lon', 'lat', 'val']]
                if 'mslp' == fieldnames[ff]:
                    FLD.lon.where( \
                        TRACKS[ii]['mslp'].flag, 1.e+25, inplace=True)
                    FLD.lat.where( \
                        TRACKS[ii]['mslp'].flag, 1.e+25, inplace=True)
                FLD.columns = ['lon%d'%(ff), 'lat%d'%(ff), 'val%d'%(ff)] 
                xTRACKS = pd.concat([xTRACKS, FLD], axis=1)
            xTRACKS.to_string(\
                outfile, \
                    formatters={'val':'{:,.6e}'.format, \
                                    'lon0':'& {:,.6e}'.format, \
                                    'lat0':'& {:,.6e}'.format, \
                                    'val0':'& {:,.6e}'.format, \
                                    'lon1':'& {:,.6e}'.format, \
                                    'lat1':'& {:,.6e}'.format, \
                                    'val1':'& {:,.6e}'.format, \
                                    'lon2':'& {:,.6e}'.format, \
                                    'lat2':'& {:,.6e}'.format, \
                                    'val2':'& {:,.6e}'.format, \
                                    'lon3':'& {:,.6e}'.format, \
                                    'lat3':'& {:,.6e}'.format, \
                                    'val3':'& {:,.6e} &'.format, \
                                    }, \
                    header=False, index_names=False)
            outfile.write('\n')

def croptracks_2sided(TRACKS, selecting_var, hours, deltat):
    import math
    import numpy as np

    track_num = len(TRACKS)
    tstep_num = int(math.floor(hours/deltat))
    for ii in range(track_num):
        ix = np.argmax(TRACKS[ii][selecting_var].val.values)
        point_num = TRACKS[ii][selecting_var].shape[0]
        if ix > tstep_num and point_num - ix - 1 >= tstep_num:
            for key in TRACKS[ii]:
                TRACKS[ii][key] = \
                    TRACKS[ii][key].iloc[ix-tstep_num:ix+tstep_num+1]
        else:
            if ix > tstep_num and point_num - ix - 1 < tstep_num:
                for key in TRACKS[ii]:
                    TRACKS[ii][key] = \
                        TRACKS[ii][key].iloc[ix-tstep_num:point_num]
            elif ix <= tstep_num and point_num - ix - 1 >= tstep_num:
                for key in TRACKS[ii]:
                    TRACKS[ii][key] = \
                        TRACKS[ii][key].iloc[0:ix+tstep_num+1]                

    return TRACKS

def selecttracks(TRACKS, indices):
    track_num = len(TRACKS)
    TRACKS = [TRACKS[x] for x in range(track_num) if indices[x]]

    return TRACKS

def readtracks_mseason(runid, season, year):
    print 'Reading tracks corresponding to %s %d in %s' % \
        ( seasons[season]['long_name'], year, runids[runid] )

    if seasons[season]['multiple_file']:
        TRACKS = []
        for ff in range(len(seasons[season]['year_offset'])):
            inputfile='ML_%d_%s_%s.full_track.%s.tracks' % \
                (year+seasons[season]['year_offset'][ff], \
                     seasons[season]['partial_season'][ff], runid, hemisphere)
            print 'Reading file: %s' % (inputfile)
            full_inputfile = trackdir+runid+'/'+inputfile
            TRACKS = TRACKS + readtracks_1season(full_inputfile)

    else:
        inputfile='ML_%d_%s_%s.full_track.%s.tracks' % \
            (year, season, runid, hemisphere)
        print 'Reading file: %s' % (inputfile)

        full_inputfile = trackdir+runid+'/'+inputfile

        TRACKS = readtracks_1season(full_inputfile)

    track_num = len(TRACKS)
    print '%d tracks loaded for %s %d' % \
        (track_num, seasons[season]['long_name'], year)

    return TRACKS

def readtracks_1season(\
    inputfile, fieldnames = ['fvort', 'wind925', 'mslp', 'wind10m'], \
    add_data_col=None, add_data_name=None):
    '''Read tracks from a single season.
    '''    

    skiprow_num = 2

    gentrackinfo = pd.read_csv(inputfile, \
                                   skiprows=skiprow_num, \
                                   delim_whitespace=True, nrows=1, header=None)

    track_num = gentrackinfo[1][0]
    add_fld  = gentrackinfo[3][0]

    print '%d tracks including %d additional fields' % (track_num, add_fld)

    cols = [0, 1, 2, 3]
    names = ['tdates', 'lon', 'lat', 'val']

    if add_data_col is not None:
        cols.append(add_data_col)
        names.append(add_data_name)

    if add_fld > 0:
        cols.extend([2*i+5 for i in range(3*add_fld)])
        for nn in range(add_fld):
            names.extend(['lon%d'%(nn+1), 'lat%d'%(nn+1), 'val%d'%(nn+1)])

    TRACKS=[]
    
    skiprow_num = skiprow_num + 2
    
    for ii in range(track_num):
        TRACKiiinfo = pd.read_csv(inputfile, \
                                      skiprows=skiprow_num, \
                                      delim_whitespace=True, \
                                      nrows=1, header=None)
        
        point_num = TRACKiiinfo[1][0]
        
        skiprow_num = skiprow_num + 1
        
        inputdata = pd.read_csv(inputfile, \
                                    skiprows=skiprow_num, \
                                    header=None, \
                                    index_col=0, \
                                    delim_whitespace=True, \
                                    nrows=point_num, \
                                    na_values=1.0E+25, \
                                    usecols=cols, \
                                    names=names)
        
        data1 = inputdata[names[1:]]

        TRACKii = { 'vort'   : data1 }
        
        if add_fld > 0:
            for nn in range(add_fld):
                data = \
                    inputdata[['lon%d'%(nn+1), 'lat%d'%(nn+1), 'val%d'%(nn+1)]]
                data.columns = ['lon', 'lat', 'val']

                TRACKii[fieldnames[nn]] = data

        if 'mslp' in fieldnames:
            TRACKii['mslp']['flag'] = ~pd.isnull(TRACKii['mslp'].lon)
            TRACKii['mslp'].lon.where( \
                TRACKii['mslp'].flag, TRACKii['vort'].lon, inplace=True)
            TRACKii['mslp'].lat.where( \
                TRACKii['mslp'].flag, TRACKii['vort'].lat, inplace=True)

        TRACKS.append(TRACKii)
        
        skiprow_num = skiprow_num + point_num + 1
    
    return TRACKS
