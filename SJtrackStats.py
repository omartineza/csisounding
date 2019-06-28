import numpy as np
import scipy.stats as sps
from scipy.io import loadmat
import netCDF4 as nc
import glob
import matplotlib.dates as mdt
import mpl_toolkits.basemap as bm
import matplotlib
Path = matplotlib.path.Path
import matplotlib.pyplot as plt
mdt = matplotlib.dates
from time import time as timer
import os
import cPickle
import matplotlib.colors as colors

def goodwindcm(*args):
    '''Registers a good colormap for rain, lightblue-to-darkblue-to-red
    returns object <matplotlib.colors.LinearSegmentedColormap>
    USAGE: don't pass anything to function call to return colormap
    if pass *anything* sensible  will return reversed version'''
    _rainbluered = ((1.0, 1.0, 1.0),(0.0, 0.0, 1.0),(1.0, 0.0, 0.2)) # one used I made in matlab
    #_rainbluered = ((1.0, 0.95, 0.3),(0.0, 1.0, 0.8),(0.0, 0.3, 1.0),(0.4, 0.0, 1.0),(1.0, 0.0, 0.2))
    LUTSIZE=256
    cmapspec = _rainbluered
    cmapname = 'rainBuRd'
    cmapname_r = 'rainBuRd_r'
    revspec = list(reversed(cmapspec))
    #pyplotmodule.cm.register_cmap(name=cmapname,data=cmapspec,lut=LUTSIZE)
    #pyplotmodule.cm.register_cmap(name=cmapname,data=cmapspec,lut=LUTSIZE)
    rainBuRd = colors.LinearSegmentedColormap.from_list(cmapname, \
                                                        cmapspec, LUTSIZE)
    rainBuRd_r = colors.LinearSegmentedColormap.from_list(cmapname_r,\
                                                          revspec, LUTSIZE)
    if len(args)==0: out = rainBuRd
    elif len(args)>0: out = rainBuRd_r
    return out

def grey_color_scale(nlevels=10,rev=False):
    '''Defines a colour scale that it colourful, not too sickly and which has
       monotonically increasing shades of grey when printed in black and white 
       so that there is no ambiguity.
       Courtesy: Peter Clark and others'''
    nnxx=nlevels-1
    R=np.hstack((np.repeat(1.0,nnxx),np.arange(1.0,0.0,-1.0/nnxx),\
                 np.arange(0.0,0.75,0.75/nnxx),np.arange(0.75,1.0,0.25/nnxx),\
                 np.arange(1.0,0.0,-1.0/nnxx),np.repeat(0.0,nnxx) ))
    G=np.hstack((np.arange(1.0,0.5,-0.5/nnxx),np.arange(0.5,1.0,0.5/nnxx),\
                 np.arange(1.0,0.75,-0.25/nnxx),np.arange(0.75,0.0,-0.75/nnxx),\
                 np.repeat(0.0,nnxx),np.repeat(0.0,nnxx) ))
    B=np.hstack((np.repeat(1.0,nnxx),np.repeat(1.0,nnxx),\
                 np.arange(1.0,0.0,-1.0/nnxx),np.repeat(0.0,nnxx),\
                 np.arange(0.0,0.5,0.5/nnxx),np.arange(0.5,0.0,-0.5/nnxx) ))
    RGB=np.hstack((R[:,np.newaxis],G[:,np.newaxis],B[:,np.newaxis]))
    gcs = colors.ListedColormap(RGB,name='grey_color_scale')
    gcs_r =colors.ListedColormap(RGB[::-1,:])
    if not rev: out = gcs
    elif rev: out = gcs_r

    return out

def spheredistance(lon1, lat1, lon2, lat2,R=6367442.76):
    '''Calculated great-circle distances in metres
       distance = spheredistance(lon1,lat1,lon2,lat2,R=6367442.76)
       USAGE: lon,lat values
              R defaults to radius of earth in metres'''
    
    # Determine proper longitudinal shift
    dlon=np.abs(lon2-lon1)
    dlon=np.where(dlon>=180,360-dlon,dlon)
    #  Convert Decimal degrees to radians.
    deg2rad=np.pi/180
    lat1=lat1*deg2rad
    lat2=lat2*deg2rad
    dlon=dlon*deg2rad
    #
    #  Compute the distances
    t1 = np.sin(dlon) * np.cos(lat2)
    t2 = np.sin(lat2) * np.cos(lat1)
    t3 = np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    dist=R*np.arcsin(np.sqrt(t1**2 + (t2-t3)**2))

    return dist

def tissotpolygon(lon_0,lat_0,radius_deg,npts=40):
    '''Return the Lon, Lat coordinates of a circular polygon of
    given radius (in latitude), on the earth's surface
    This method is taken straight out of '''
    g = bm.pyproj.Geod(a=6370997.0,b=6370997.0)
    az12,az21,dist = g.inv(lon_0,lat_0,lon_0,lat_0+radius_deg)
    seg = [(lon_0,lat_0+radius_deg)]
    delaz = 360./npts
    az = az12
    for n in range(npts):
        az = az+delaz
        lon, lat, az21 = g.fwd(lon_0, lat_0, az, dist)
        seg.append((lon,lat))
    return np.asarray(seg)

def valuedensity(gridlons,gridlats,lonlatpoints,angulardistance=2.):
    density = np.zeros((len(gridlons),len(gridlats)))
    density[:] = np.nan
    for ix, glon in enumerate(gridlons):
        for iy, glat in enumerate(gridlats):
            cpoly = tissotpolygon(glon,glat,angulardistance,npts=20)
            cpath = Path(cpoly)
            inpoly = cpath.contains_points(lonlatpoints)
            density[ix,iy]=inpoly.sum()
    return density

def trackdensity(gridlons, gridlats, trklist, angulardistance=2., \
                     dataset='erai'):
    density = np.zeros((len(gridlons),len(gridlats)))
    density[:] = np.nan
    for ix, glon in enumerate(gridlons):
        for iy, glat in enumerate(gridlats):
            cpoly = tissotpolygon(glon,glat,angulardistance,npts=20)
            cpath = Path(cpoly)
            inpoly=0
            for trk in trklist:
                if dataset == 'erai':
                    lonlatpoints=np.hstack((trk.vlon,trk.vlat))
                elif dataset == 'ajthr' or dataset == 'ajthp':
                    lonlatpoints = \
                        np.transpose(np.vstack((trk.vlon,trk.vlat)))
                trackinpoly = cpath.contains_points(lonlatpoints)
                if np.any(trackinpoly): inpoly+=1
            density[ix,iy]=inpoly
    return density

def winddensity(gridlons,gridlats,trklist,angulardistance=2.):
    density = np.zeros((len(gridlons),len(gridlats)))
    density[:] = np.nan
    for ix, glon in enumerate(gridlons):
        for iy, glat in enumerate(gridlats):
            cpoly = tissotpolygon(glon,glat,angulardistance,npts=20)
            cpath = Path(cpoly)
            inpoly=0
            for trk in trklist:
                lonlatpoints=np.hstack((trk.vlon,trk.vlat))
                trackinpoly = cpath.contains_points(lonlatpoints)
                if np.any(trackinpoly): inpoly+=1
            density[ix,iy]=inpoly
    return density

def windswathdensity(gridlons, gridlats, trklist, windthresh=30):
    Ntracks = len(trklist)
    Jlat = len(gridlats)
    Ilon = len(gridlons)
    swathbool=np.bool8( np.zeros((Ntracks, Jlat, Ilon)) )
    swathboolc=np.bool8( np.zeros((Ntracks, Jlat, Ilon)) )
    swathboolw=np.bool8( np.zeros((Ntracks, Jlat, Ilon)) )
    for i, trk in enumerate(trklist):
        swathbool[i,:] = trk.uvfootprint > windthresh
        swathboolc[i,:] = trk.uvfootprintc > windthresh
        swathboolw[i,:] = trk.uvfootprintw > windthresh
    return swathbool.sum(0), swathboolc.sum(0), swathboolw.sum(0)

def windswathdensity_areamask(gridlons, gridlats, trklist, areamask, \
                                  windthresh=30):
    Ntracks = len(trklist)
    Jlat = len(gridlats)
    Ilon = len(gridlons)
    swathbool=np.bool8( np.zeros((Jlat, Ilon)) )
    Nstorm = 0
    Nnostorm = 0
    for i, trk in enumerate(trklist):
        swathbool = np.logical_and(trk.uvfootprint > windthresh, areamask)
        if np.any(swathbool):
            Nstorm+=1
        else:
            Nnostorm+=1

    return Nstorm
    
def axistick2nd(axis,tickarr):
    if isinstance(tickarr[0],matplotlib.text.Text):
        tickpos = [tx.get_position() for tx in tickarr]
        labels = [tx.get_text() for tx in tickarr]
        #tickarr=np.asarray(tickpos)[:,0ti]
        exec('tickarr=['+','.join(labels)+']')
    labels=[str(d) for d in tickarr[1::2]]
    labels=','+(',,').join(labels)+','
    labels=labels.split(',')
    if len(labels)>len(tickarr):labels=labels[:-1]
    exec('plt.'+axis+'ticks(tickarr,labels)')

def ytickfonts(fontweight='bold',fontsize=13.,rotation=False):
    ax=plt.gca()
    labels=[l for l in ax.yaxis.get_ticklabels()]
    for lb in labels:
        lb.set_fontsize(fontsize)
        lb.set_fontweight(fontweight)
        if rotation: lb.set_rotation(rotation)

def xtickfonts(fontweight='bold',fontsize=13.,rotation=False):
    ax=plt.gca()
    labels=[l for l in ax.xaxis.get_ticklabels()]
    for lb in labels:
        lb.set_fontsize(fontsize)
        lb.set_fontweight(fontweight)
        if rotation: lb.set_rotation(rotation)

def NAtlortho(lat0=45,lon0=-25,resolution='c'):
    m1 = bm.Basemap(projection='ortho', lon_0=lon0, lat_0=lat0, resolution='c')
    m = bm.Basemap(projection='ortho', lon_0=lon0, lat_0=lat0,\
        resolution=resolution,\
        llcrnrx=-m1.urcrnrx/3, llcrnry=-m1.urcrnry/8,\
        urcrnrx=m1.urcrnrx/3, urcrnry=m1.urcrnry/4.5)
    return m

def drawsubdomain(m, subdomain, colour, linestyle):
    ln1,ln2,lt1,lt2 = subdomain
    s1 = np.asarray((np.tile(ln1, (50)), np.linspace(lt1, lt2)))
    s2 = np.asarray((np.linspace(ln1, ln2), np.tile(lt2, (50))))
    s3 = np.asarray((np.tile(ln2, (50)), np.linspace(lt2, lt1)))
    s4 = np.asarray((np.linspace(ln2, ln1), np.tile(lt1, (50))))
    sq = np.hstack((s1, s2, s3, s4))
    ### Project region on to map
    xs,ys=m(sq[0,:], sq[1,:])
    ### Deal with spurios values where line outside of map vision
    nanny=np.tile(np.nan, 200)
    xs=np.where(xs>1.0e+15, nanny, xs)
    m.plot(xs, ys, lw=2, color=colour, ls=linestyle)

def drawNAtlsubdomain(m, linestyle='--', linewidth=2):
    ### Draw selection regions if desired
    ### Region definition
    lon1 = -130;lon2 = 60;
    lat1 = 0;lat2 = 89;
    bigregion=(lon1,lon2,lat1,lat2,'k')
    ### Maximum rel vort region definition
    lon1_mv = -80;lon2_mv = 40;
    lat1_mv = 45;lat2_mv = 70;
    vortregion=(lon1_mv,lon2_mv,lat1_mv,lat2_mv,'k')
    ### Start region definition
    lon1_s = -130;lon2_s = 10;
    lat1_s = 20;lat2_s = 70;
    startregion=(lon1_s,lon2_s,lat1_s,lat2_s,'b')
    if True:
        #for reg in (bigregion,vortregion,startregion,):
        for reg in (vortregion,):
            ln1,ln2,lt1,lt2,clr=reg;#ln1=360+ln1
            s1=np.asarray((np.tile(ln1,(50)),np.linspace(lt1,lt2)))
            s2=np.asarray((np.linspace(ln1,ln2),np.tile(lt2,(50))))
            s3=np.asarray((np.tile(ln2,(50)),np.linspace(lt2,lt1)))
            s4=np.asarray((np.linspace(ln2,ln1),np.tile(lt1,(50))))
            sq=np.hstack((s1,s2,s3,s4))
            ### Project region on to map
            xs,ys=m(sq[0,:],sq[1,:])
            ### Deal with spurios values where line outside of map vision
            nanny=np.tile(np.nan,200)
            xs=np.where(xs>1.0e+15,nanny,xs )
            m.plot(xs,ys,lw=linewidth,color=clr,ls=linestyle)

def drawstuff(m, coastlinecolor='.5', labellon=[0,0,0,1], labellat=[0,1,0,0], \
                  fontweight='bold'):
    m.drawparallels(np.arange(10,85,10),labels=labellat,\
                    fontsize=14,fontweight=fontweight)
    m.drawmeridians(np.arange(-180,180,20),labels=labellon,\
                    fontsize=14,fontweight=fontweight)
    m.drawcoastlines(color=coastlinecolor)

posstatvars=('trkno', 'ixt', 'minr', 'meanr', 'maxr', 'minalpha', 'alpha',\
             'maxalpha', 'mlon_ixt', 'mlat_ixt', 'mlon_ixt1', 'mlat_ixt1',\
             'chi', 'minbeta', 'beta', 'maxbeta', 'mine', 'meane', 'maxe',\
             'minp', 'meanp', 'maxp')

class CycloneTrack:
    '''CycloneTrack class containing a the lat, lon coords of a cyclone track
       and a number of additional along track properties'''
    def __init__(self, oddloadmattrack, selfno=False, number=None, \
                     inseason_number=None):
        '''This can be modified as you need. Currently it takes in one
        track from a slightly obscure struct which scipy.io.loadmat returns
        from the loaded mat-file. These tracks are from OMA's sjprecursor
        tracks'''
        mattrk = oddloadmattrack
        self.sjstorm=False
        ### This is currently hard-coded for the specifics of these tracks
        valstr=('trkno','nsteps','tdates','vlon','vlat','vort','mdates',\
                'mix','mlon','mlat','mslp','csiday','csihour')
        if len(mattrk)==13:
            for n in xrange(13):
                exec('self.%s = mattrk[n]' %(valstr[n]))
                #exec('self.%s = np.float64(mattrk[n])' %(valstr[n]))
            self.csiday=np.where(self.csiday[0]==0,np.nan,self.csiday[0])
            self.csihour=np.where(np.isnan(self.csiday),np.nan,self.csihour[0])
            self.trkno = self.trkno[0][0]
            self.nsteps = self.nsteps[0][0]
            self.mix = self.mix[0]
            self.nt = len(self.tdates) 
            self.imxv = self.vort.argmax()
            self.mxv = self.vort.max()
        elif len(mattrk)==11:
            for n in xrange(11):
                exec('self.%s = mattrk[n]' %(valstr[n]))
                #exec('self.%s = np.float64(mattrk[n])' %(valstr[n]))
            self.trkno = self.trkno[0][0]
            self.nsteps = self.nsteps[0]
            self.mix = self.mix[0]
            self.nt = len(self.tdates) 
        elif len(mattrk)==10:
            valstr=('str_start', 'nsteps', 'tdates', 'vort', \
                        'fvort', 'wind925','mslp','wind10m','csiday','csihour')
            for n in xrange(10):
                exec('self.%s = mattrk[n]' % (valstr[n]))
            self.mdates = self.tdates[0]
            self.csiday=np.where(self.csiday[0]==0,np.nan,self.csiday[0])
            self.csihour=np.where(np.isnan(self.csiday),np.nan,self.csihour[0])
            self.nsteps = self.nsteps[0]
            self.nt = len(self.tdates[0])
            self.vlon = self.vort[0][0][0][0]
            self.vlat = self.vort[0][0][1][0]
            self.vort = self.vort[0][0][2][0]
            self.mlon = self.mslp[0][0][0][0]
            self.mlat = self.mslp[0][0][1][0]
            self.mslp = self.mslp[0][0][2][0]
            self.imxv = self.vort.argmax()
            self.mxv = self.vort.max()
            if selfno:
                self.trkno = number
                self.inseasonno = inseason_number
        else:
            print "Track doesn't have 10, 11 or 13 properties. " + \
                "I don't know what to do!"
            return False

        if len(self.mslp)>4:
            dp24 = self.mslp[4:] - self.mslp[:-4]
            self.mslpdrop24 = dp24
            avglat=(self.mlat[4:]+self.mlat[:-4])/2
            latscale = np.sin(np.deg2rad(avglat)) / np.sin(np.deg2rad(60))
            self.mslpdrop24_scaled = dp24*latscale
        else:
            self.mslpdrop24, self.mslpdrop24_scaled = np.nan, np.nan 

### THIS BELOW IS THE LIST OF COLUMN HEADERS IN POSSTATS ARRAY
###trkno ixt minr meanr maxr minalpha alpha maxalpha mlon_ixt mlat_ixt\
###mlon_ixt1 mlat_ixt1 chi minbeta beta maxbeta mine meane maxe minp meanp maxp
def mattracks2dict_csi(tracks,selftrkno=False,numstart=None):
    TRACKS={}
    if not selftrkno:
        Ntracks = len(tracks[0])
    else:
        Ntracks = len(tracks)
    for j in xrange(Ntracks):
        if not selftrkno:
            mattrack = tracks[0][j]
            trkno = np.int64(mattrack[0][0][0])
            TRACKS[trkno] = CycloneTrack(mattrack)
        else:
            mattrack = tracks[j][0]
            trkno = j+numstart
            TRACKS[trkno] = CycloneTrack( \
                mattrack, selfno=True, number=trkno, inseason_number=j+1)
    return TRACKS    
    

def filtertracks(posstats, sjpotential, dispmsg=False, maxradius=700e3, \
                     min_csipts=8, blob_azimuth_thresh=(-60,150) ):
    OneEighty=180; ThreeSixty=360
    az1, az2 = blob_azimuth_thresh
    weight_vert = False
    rowno = len(sjpotential)
    if rowno != posstats.shape[0]:
        print 'WARNING: POSSTATS contains too many rows'
    cyccount = 0
    invalid = np.zeros((rowno,1))
    bloblen = np.zeros((rowno,1))
    inwcb  = np.zeros((rowno,1))
    validcsipts, invalidcsipts = [], [] 
    for ix in xrange(rowno):
        pstar = sjpotential[ix]['pstar'][0]
        alpha = sjpotential[ix]['alpha'][0]
        chi = sjpotential[ix]['chi'][0]
        r = sjpotential[ix]['r'][0]
        energy = sjpotential[ix]['energy'][0]

        if ix == 0:
            if dispmsg:
                print 'Cyclone: %d' % (posstats[ix,0])
            count = 0;count1 = 0;count2 = 0;cyccount += 1;
        else:
            if posstats[ix,0] != posstats[ix-1,0]:
                if dispmsg:
                    print str(count), ' blobs'
                if count1>0:
                    if dispmsg:
                        print str(count1), ' blobs are very short'
                    if count1 == count:
                        if dispmsg:
                            print 'Consider discarding this cyclone'
                if count2>0:
                    if dispmsg:
                        print str(count2), ' blobs are in WCB'
                    if count2 == count:
                        if dispmsg:
                            print 'Consider discarding this cyclone'
                if dispmsg:
                    print 'Cyclone: %d' % (posstats[ix,0])
                count = 0;count1 = 0;count2 = 0;
                cyccount = cyccount + 1;
        wpoint = np.ones(pstar.shape)
        if weight_vert:
            wpoint[(pstar==750).nonzero()[0]] = 0.75
            wpoint[(pstar==775).nonzero()[0]] = 0.5
            wpoint[(pstar==800).nonzero()[0]] = 0.5
        wlen = np.sum(wpoint)
        if wlen < min_csipts: # Is ix-th blob too small?
            invalid[ix] = 1;
            bloblen[ix] = wlen
            count1 += 1
        blob_azimuth = alpha - chi + 180
        if np.all((blob_azimuth > az1) & (blob_azimuth < az2)) and \
                np.all(r>=250): # Is ix-th blob in WCB?
            inwcb[ix] = 1
            count2 += 1
        else:
            validcsipts.append( (posstats[ix,0],wlen) )	
        count += 1
    ixsj = np.where( (inwcb+invalid)==0 )[0]
    sjtrackids = np.unique(posstats[ixsj,0])
    validcsipts = np.asarray(validcsipts)
    numcsipts=[]
    for sjid in sjtrackids:
        ixno = np.where(validcsipts[:,0]==sjid)[0]
        if len(ixno)>1: numcsipts.append(validcsipts[ixno,1].max())
        else: numcsipts.append(validcsipts[ixno[0],1])
    ### Failure Diagnostics
    ixnosj = np.where( (inwcb+invalid)>0 )[0]
    probnosjtrackids = np.unique(posstats[ixnosj,0])
    failed=[];faillen=np.nan
    for fid in probnosjtrackids:
        if np.any(sjtrackids==fid):continue
        ixno = np.where(posstats[:,0]==fid)[0]
        failreason=0
        for ct, ixn in enumerate(ixno):
            freason=0
            if inwcb[ixn]==1:
                freason=7
            elif invalid[ixn]==1:
                freason==9
            if freason>failreason: 
                failreason, faillen = freason, bloblen[ixn]
        failed.append( (fid,failreason,faillen) )

    return ixsj, sjtrackids, np.asarray(failed), numcsipts 

def addprecusorstats2sjtracks( \
    tracks, posstats,sjpotential, \
        blob_azimuth_thresh=(-60,100), min_csipts=8):
    '''Call filter tracks and then add sj precusor properties to
    tracks that pass as having stingjet potential'''
    ixsj, sjtrackids, failed, numcsipts = filtertracks( \
        posstats, sjpotential, \
            blob_azimuth_thresh=blob_azimuth_thresh, min_csipts=min_csipts)
    if len(sjtrackids) != len(numcsipts): print 'Uh oh!'
    if len(failed.shape) > 1:
        for num,trkid in enumerate(failed[:,0]):
            tracks[trkid].failreason=failed[num,1]
            tracks[trkid].blobfaillen=failed[num,2]
        
    ### Clear previous sjstorm assigments
    for trkid in tracks.keys(): tracks[trkid].sjstorm=False
    for itr,trkid in enumerate(sjtrackids):
        tracks[trkid].sjstorm=True
        tracks[trkid].maxnumcsipts = numcsipts[itr]
        tracks[trkid].csiblob_varstrs=posstatvars[2:]
        trkixs = np.where(trkid==posstats[ixsj,0])[0]
        nt=tracks[trkid].nt
        nc=posstats.shape[1]-2
        nanarr=np.zeros((nt,nc));nanarr[:]=np.nan
        tracks[trkid].csiblob_stats = nanarr
        tracks[trkid].ixt_csiblobs = posstats[ixsj[trkixs],1]-1 
        # data original created in matlab hence -1 needed for index values
        if len(trkixs)==1:
            tracks[trkid].sjpots = {}
            trkix=trkixs[0]
            ixposs = ixsj[trkix]
            pstar = sjpotential[ixposs]['pstar'][0]
            alpha = sjpotential[ixposs]['alpha'][0]
            chi = sjpotential[ixposs]['chi'][0]
            r = sjpotential[ixposs]['r'][0]
            energy = sjpotential[ixposs]['energy'][0]
            sjpot = {'pstar':pstar,'alpha':alpha,\
                     'chi':chi,'r':r,'energy':energy}
            ixt = posstats[ixposs,1]-1
            tracks[trkid].sjpots[ixt]=sjpot
            tracks[trkid].csiblob_stats[ixt,:]=posstats[ixposs,2:]
        elif len(trkixs)>1:
            tracks[trkid].sjpots = {}
            for trkix in trkixs:
                ixposs = ixsj[trkix]
                ixt = posstats[ixposs,1]-1
                pstar = sjpotential[ixposs]['pstar'][0]
                alpha = sjpotential[ixposs]['alpha'][0]
                chi = sjpotential[ixposs]['chi'][0]
                r = sjpotential[ixposs]['r'][0]
                energy = sjpotential[ixposs]['energy'][0]
                sjpot = {'pstar':pstar,'alpha':alpha,\
                         'chi':chi,'r':r,'energy':energy}
                ixt = posstats[ixposs,1]-1
                tracks[trkid].sjpots[ixt]=sjpot
                tracks[trkid].csiblob_stats[ixt,:]=posstats[ixposs,2:]
        else:
            print "Oh no! problem!"
            return False
    return sjtrackids, numcsipts

def trkassignERAdata(trk,stormdata,filepattern,commongrid,\
                     varlist=['U','V','SJMASK','THW']):
    monthstr=['jan','feb','mar','apr','may','jun',\
              'jul','aug','sep','oct','nov','dec']
    lns,lts=commongrid
    gridshape=lns.shape
    gpts=np.hstack((lns.ravel()[:,np.newaxis],lts.ravel()[:,np.newaxis]))
    #gpts=np.vstack((lns.ravel(),lts.ravel()))
    trkdates=trk.mdates
    ndt = len(trkdates)
    imv = trk.imxv
    trk.thetaw99=np.zeros((ndt))
    if 'U' in varlist:
        trk.uvmax=np.zeros((ndt))
        trk.uvmaxw=np.zeros((ndt))
        trk.uvmaxc=np.zeros((ndt))
        trk.uvmag=np.zeros((ndt,gridshape[0],gridshape[1]))
        trk.uvmagw=np.zeros((ndt,gridshape[0],gridshape[1]))
        trk.uvmagc=np.zeros((ndt,gridshape[0],gridshape[1]))
    for n in xrange(ndt):
        trkno=np.int64(trk.trkno)
        lon0m,lat0m=trk.mlon[n][0],trk.mlat[n][0]
        polypts= tissotpolygon(lon0m,lat0m,10.,npts=20) 
        cpoly = Path(polypts)
        inpoly = cpoly.contains_points(gpts)
        dtm=str(trkdates[n])
        yr,mm,dd,hr = int(dtm[:4]),int(dtm[4:6]),int(dtm[6:8]),int(dtm[8:])
        mn=monthstr[mm-1]
        trkdatetime=np.int64(dtm)
        exec('erafile='+filepattern)
        if not os.path.isfile(erafile):
            print 'Oops,',erafile,'missing'
            continue
        ncf=nc.Dataset(erafile,'r')
        lat = ncf.variables['latitude'][:]
        lon = ncf.variables['longitude'][:]
        ln,lt = np.meshgrid(lon,lat)
        vargridshape=ln.shape
        vargpts = np.hstack((ln.ravel()[:,np.newaxis],lt.ravel()[:,np.newaxis]))
        varinpoly = cpoly.contains_points(vargpts)
        if varinpoly.sum()==inpoly.sum():
            coords=np.reshape(inpoly,gridshape).nonzero()
            varcoords=np.reshape(varinpoly,vargridshape).nonzero()
        else:
            print erafile
            print "This isn't working...",inpoly.sum(),varinpoly.sum()
            continue
            #return False
        plev = ncf.variables['p'][:]
        iplev = np.where(plev==850)[0][0]
        if 'SJMASK' in varlist:
            sjmask = ncf.variables['SJMASK'][:].squeeze()
            dscapesj=np.ma.MaskedArray(sjmask,mask=~np.bool8(sjmask))
        if 'DSCAPE' in varlist:
            dscape = ncf.variables['DSCAPE'][:]
            dscapem=dscape.max(0)
        if 'THW' in varlist:
            thetaw = ncf.variables['THW'][:][iplev,:,:]
            maggrd = ncf.variables['MAGGRAD'][:][0,:,:]
            inz = np.where(maggrd>0) ### This will just get threshold
            ### Alternatively ensure threshold only found from
            ### gradients south of cyclone centre
            #inz = np.where( (maggrd>0) & (lt<lat0m) )
            perc = np.percentile(maggrd[inz],99)
            ipc = np.where(maggrd>perc)
            thetathresh = thetaw[ipc].mean()
            warmsector = thetaw>thetathresh
            coolsector = thetaw<thetathresh
            trk.thetaw99[n] = thetathresh
        if 'MSLP' in varlist: mslp = ncf.variables['MSLP'][:].squeeze()
        if 'U' in varlist: u = ncf.variables['U'][iplev,:,:]
        if 'V' in varlist: 
            v = ncf.variables['V'][iplev,:,:]
            uvmag = np.sqrt(u**2 + v**2)
            uvmagw = np.where(warmsector,uvmag,0.)
            uvmagc = np.where(coolsector,uvmag,0.)
            trk.uvmagc[n,coords[0],coords[1]]=uvmagc[varcoords]
            trk.uvmagw[n,coords[0],coords[1]]=uvmagw[varcoords]
            trk.uvmag[n,coords[0],coords[1]]=uvmag[varcoords]
            trk.uvmax[n]=uvmag[varcoords].max()
            trk.uvmaxw[n]=uvmagw[varcoords].max()
            trk.uvmaxc[n]=uvmagc[varcoords].max()
        ncf.close()
    trk.uvfootprint=np.ma.MaskedArray(trk.uvmag.max(0),\
                                      mask=(trk.uvmag.max(0)==0))
    trk.uvfootprintw=np.ma.MaskedArray(trk.uvmagw.max(0),\
                                      mask=(trk.uvmagw.max(0)==0))
    trk.uvfootprintc=np.ma.MaskedArray(trk.uvmagc.max(0),\
                                      mask=(trk.uvmagc.max(0)==0))
    #trk.footprintlonslats = (lns,lats)
    trk.uvmag="removed"
    trk.uvmagw="removed"
    trk.uvmagc="removed"
    #trk.uvmag = np.where(trk.uvmag==0,np.nan,trk.uvmag)
    return trk

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

def fplane_cartesian_grid(EQLON, EQLAT):
    from planetary_constants import a
    
    X = EQLON
    Y = EQLAT
    
    X1 = spheredistance(EQLON, EQLAT, np.zeros_like(EQLON), EQLAT, R=a)
    Y1 = spheredistance(EQLON, EQLAT, EQLON, np.zeros_like(EQLAT), R=a)

    X = np.where(EQLON<0, -X1, X1)
    Y = np.where(EQLAT<0, -Y1, Y1)

    return X, Y

def find_central_contour(contour_line, points, central_points):
    errflag = True
    for ii in np.arange(len(contour_line)):
        contour_ii = Path(contour_line[ii])
        incontour = contour_ii.contains_points(points)
        if np.any(np.logical_and(incontour, central_points)):
            icentral = ii
            errflag = False

    if errflag:
        icentral = np.nan

    return icentral, errflag
        

def trkassigndatasetdata(trk, stormdata, filepattern, commongrid, \
                             varlist=['U', 'V', 'SJMASK', 'THW']):
    import numpy.ma as ma
    from datetime import datetime
    from scipy.interpolate import griddata
    from planetary_constants import a
    import tropopause_contour as tropoc

    # import plotformatting as pltfmt
    # import matplotlib.cm as mpl_cm

    leap = range(1992, 2020, 4)

    lns,lts=commongrid
    gridshape=lns.shape
    gpts=np.hstack((lns.ravel()[:,np.newaxis], lts.ravel()[:,np.newaxis]))
    trkdates=trk.mdates
    ndt = len(trkdates)
    imv = trk.imxv
    trk.thetaw99=np.zeros((ndt))

    if 'MSLP' in varlist:
        trk.eqradius = np.zeros((ndt))

    if 'U' in varlist:
        trk.uvmax=np.zeros((ndt))
        trk.uvmaxw=np.zeros((ndt))
        trk.uvmaxc=np.zeros((ndt))
        trk.uvmag=np.zeros((ndt,gridshape[0],gridshape[1]))
        trk.uvmagw=np.zeros((ndt,gridshape[0],gridshape[1]))
        trk.uvmagc=np.zeros((ndt,gridshape[0],gridshape[1]))

    for n in xrange(ndt):
        trkno=np.int64(trk.inseasonno)
        true_lon0m, true_lat0m = trk.mlon[n], trk.mlat[n]
        lonp = true_lon0m + 180;
        latp = 90 - true_lat0m;
        lon0m = 0. 
        lat0m = 0.
        polypts= tissotpolygon(lon0m, lat0m, 10., npts=20) 
        cpoly = Path(polypts)
        inpoly = cpoly.contains_points(gpts)
        dtm = str(trkdates[n][0])
        yr, mm, dd, hr = \
            int(dtm[:4]), int(dtm[4:6]), int(dtm[6:8]), int(dtm[8:])
        if mm >= 3 and mm < 6:
            season = 'mam'
        elif mm >=6 and mm < 9:
            season = 'jja'
        elif mm >= 9 and mm < 12:
            season = 'son'
        else:
            season = 'djf'
        if season == 'djf' and mm == 12:
            yr = yr + 1
        # DTM HAS TO BE ADJUSTED TO ACCOUNT FOR 360-DAY YEARS
        if mm == 2 and dd > 28 and not (yr in leap):
            mm = 3
            dd = dd - 28
            d = datetime(yr, mm, dd, hr, 00)
            dtm = d.strftime('%Y%m%d%H%M')
        elif mm == 2 and dd > 29 and yr in leap:
            mm = 3
            dd = dd - 29
            d = datetime(yr, mm, dd, hr, 00)
            dtm = d.strftime('%Y%m%d%H%M')
        trkdatetime=np.int64(dtm)
        exec('erafile='+filepattern)
        # print 'Reading %s' % (erafile)
        if not os.path.isfile(erafile):
            print 'Oops, %s missing' % (erafile)
            continue
        ncf=nc.Dataset(erafile,'r')
        lat = ncf.variables['latitude'][:]
        lon = ncf.variables['longitude'][:]
        ln, lt = np.meshgrid(lon, lat)
        vargridshape=ln.shape
        vargpts = np.hstack((ln.ravel()[:,np.newaxis],lt.ravel()[:,np.newaxis]))
        varinpoly = cpoly.contains_points(vargpts)
        varcoords = np.reshape(varinpoly, vargridshape).nonzero()
        plev = ncf.variables['p'][:]
        iplev = np.where(plev==850)[0][0]
        if 'SJMASK' in varlist:
            sjmask = ncf.variables['SJMASK'][:].squeeze()
            dscapesj=np.ma.MaskedArray(sjmask,mask=~np.bool8(sjmask))
        if 'DSCAPE' in varlist:
            dscape = ncf.variables['DSCAPE'][:]
            dscapem=dscape.max(0)
        if 'THW' in varlist:
            thetaw = ncf.variables['THW'][:][:,:,iplev]
            maggrd = ncf.variables['MAGGRAD'][:,:,0]
            inz = np.where(maggrd>0) ### This will just get threshold
            ### Alternatively ensure threshold only found from
            ### gradients south of cyclone centre
            #inz = np.where( (maggrd>0) & (lt<lat0m) )
            perc = np.percentile(maggrd[inz],97)
            ipc = np.where(maggrd>perc)
            thetathresh = thetaw[ipc].mean()
            warmsector = thetaw>thetathresh
            coolsector = thetaw<thetathresh
            trk.thetaw99[n] = thetathresh
        if 'MSLP' in varlist: 
            mslp = ncf.variables['MSLP'][:].squeeze()     
            LAMBDA, PHI =  true_gcoords(lon, lat, lonp, latp)
            EQLAT, EQLON = np.meshgrid(lat, lon)
            X, Y = fplane_cartesian_grid(EQLON, EQLAT)
            points = np.array([X.flatten(), Y.flatten()]).T
            central_points = np.logical_and( \
                np.absolute(X.flatten())<100000., \
                    np.absolute(Y.flatten())<100000.)
            delta_lat = lat[1] - lat[0]
            delta_lon = lon[1] - lon[0]
            nabla_mslp_x, nabla_mslp_y = \
                np.gradient(mslp, np.radians(delta_lon), np.radians(delta_lat))
            cosPHI = np.cos(np.radians(EQLAT))
            darea = a**2*np.radians(delta_lon)*np.radians(delta_lat)* \
                cosPHI.flatten()
            nabla_mslp_y = np.divide(nabla_mslp_y, cosPHI)
            nabla_mslp_x = nabla_mslp_x/a
            nabla_mslp_y = nabla_mslp_y/a
            r_nabla_mslp_r = X*nabla_mslp_x + Y*nabla_mslp_y
            if 'DIST' in varlist:
                dist = ncf.variables['DIST'][:].squeeze()
            else:
                dist = np.sqrt(X**2 + Y**2)
            masked_mslp = ma.masked_where( \
                np.logical_or( \
                    dist < 250e3, \
                        np.absolute(r_nabla_mslp_r) > 0.5), mslp).compressed()
            masked_dist = ma.masked_where( \
                np.logical_or( \
                    dist < 250e3, \
                        np.absolute(r_nabla_mslp_r) > 0.5), dist).compressed()
            if masked_dist.size > 0:
                mslp_closed_contour = masked_mslp[masked_dist.argmin()]
                errflag = False
                errflag2 = False
                closed_contour_found = False
            else:
                errflag = False
                errflag2 = True
                closed_contour_found = False

            while not errflag and not errflag2 and not closed_contour_found:
                fig = plt.figure()
                cs = plt.contour(X, Y, mslp, [mslp_closed_contour], \
                                     color='black', linewidth=2)
                contours = tropoc.get_contour_verts(cs)
                if len(contours[0]) > 0:
                    icentral, error_cc = \
                        find_central_contour(contours[0], \
                                                 points, \
                                                 central_points)
                    if not error_cc:
                        central_contour = contours[0][icentral]
                        if tropoc.is_closed_contour(central_contour):
                            closed_contour_found = True
                        else:
                            mslp_closed_contour -= 1.
                    else:
                        errflag2 = True
                else:
                    central_contour = np.empty([1,2])
                    errflag = True
                plt.close(fig)

            if errflag2:
                dim_mslp = np.where( dist < 1000e3, mslp, np.nan)
                central_mslp = np.min(mslp.flatten()[central_points])
                mslp_closed_contour = central_mslp + 1.
                errflag2 = False
                closed_contour_found = False
                last_closed_contour_found = False
                while not errflag2 and not last_closed_contour_found:
                    fig = plt.figure()
                    cs = plt.contour(X, Y, dim_mslp, \
                                         [mslp_closed_contour], \
                                         color='black', linewidth=2)
                    contours = tropoc.get_contour_verts(cs)
                    if len(contours[0]) > 0:
                        imax, len_max = tropoc.find_longest_contour( \
                            contours[0])
                        central_contour = contours[0][imax]
                        if tropoc.is_closed_contour(central_contour):
                            closed_contour_found = True
                            previous_mslp_closed_contour = mslp_closed_contour
                            previous_central_contour = central_contour
                            mslp_closed_contour += 1.
                        else:
                            last_closed_contour_found = True
                            if closed_contour_found:
                                mslp_closed_contour = \
                                    previous_mslp_closed_contour
                                central_contour = previous_central_contour
                            else:
                                errflag2 = True
                    else:
                        central_contour = np.empty([1,2])
                        errflag2 = True
                    plt.close(fig)

            if not errflag and not errflag2 and closed_contour_found:
                mslp_contour = Path(central_contour)
                incontour = mslp_contour.contains_points(points)
                area_total = np.sum(darea)
                cyclone_area = np.sum(darea[incontour])
                eqradius = np.sqrt(cyclone_area/np.pi)
            else:
                print 'Contour search failed.'
                eqradius = 0
            
            trk.eqradius[n] = eqradius

        if 'U' in varlist: u = ncf.variables['U'][:, :, iplev]
        if 'V' in varlist: 
            v = ncf.variables['V'][:,:,iplev]
            uvmag = np.sqrt(u**2 + v**2)
            uvmagw = np.where(warmsector,uvmag,0.)
            uvmagc = np.where(coolsector,uvmag,0.)
            uvmag_t = np.zeros_like(uvmag)
            uvmagw_t = np.zeros_like(uvmag)
            uvmagc_t = np.zeros_like(uvmag)
            uvmagc_t[varcoords[0], varcoords[1]] = uvmagc[varcoords]
            uvmagw_t[varcoords[0], varcoords[1]] = uvmagw[varcoords]
            uvmag_t[varcoords[0],varcoords[1]] = uvmag[varcoords]
            trk.uvmax[n]=uvmag[varcoords].max()
            trk.uvmaxw[n]=uvmagw[varcoords].max()
            trk.uvmaxc[n]=uvmagc[varcoords].max()
            LAMBDA, PHI =  true_gcoords(lon, lat, lonp, latp)
            trk.uvmag[n, :, :] = griddata((LAMBDA.flatten(), PHI.flatten()), \
                                              uvmag_t.flatten(), \
                                              (lns, lts), method='linear', \
                                              fill_value=0.)
            trk.uvmagw[n, :, :] = griddata((LAMBDA.flatten(), PHI.flatten()), \
                                               uvmagw_t.flatten(), \
                                               (lns, lts), method='linear', \
                                               fill_value=0.)
            trk.uvmagc[n, :, :] = griddata((LAMBDA.flatten(), PHI.flatten()), \
                                               uvmagc_t.flatten(), \
                                               (lns, lts), method='linear', \
                                               fill_value=0.)
            theta_on_common = griddata((LAMBDA.flatten(), PHI.flatten()), \
                                               thetaw.flatten(), \
                                               (lns, lts), method='linear', \
                                               fill_value=0.)

            '''
            cmap = pltfmt.cmap_discretize(mpl_cm.get_cmap('BuGn'), 5)
            plt.figure(1)
            plt.clf()
            plt.contourf(lns, lts, trk.uvmag[n,:, :], \
                             np.linspace(20., 45., 6), cmap=cmap, \
                             extend='max')
            plt.colorbar()
            plt.contour(lns, lts, theta_on_common, \
                            np.linspace(270., 370., 21), colors='black')
            plt.plot(true_lon0m, true_lat0m, marker='+')
            plt.show()
            plt.close('all')
            '''
        ncf.close()
    trk.uvfootprint=np.ma.MaskedArray(trk.uvmag.max(0),\
                                      mask=(trk.uvmag.max(0)==0))
    trk.uvfootprintw=np.ma.MaskedArray(trk.uvmagw.max(0),\
                                      mask=(trk.uvmagw.max(0)==0))
    trk.uvfootprintc=np.ma.MaskedArray(trk.uvmagc.max(0),\
                                      mask=(trk.uvmagc.max(0)==0))
    trk.uvmag="removed"
    trk.uvmagw="removed"
    trk.uvmagc="removed"
    return trk

def savetrackdict(fname,trackdict):
    pickf = open(fname,'w')
    timerstart=timer()
    cPickle.dump(trackdict,pickf)
    pickf.close()
    print 'Saved to %s in %5.3f s' % (fname, timer()-timerstart)

def opentrackdict(fname):
    pickf = open(fname,'r')
    trackdict = cPickle.load(pickf)
    pickf.close()
    return trackdict

def nao_seasonindex(filename=\
    '/home/rb904381/sjcode/norm.nao.monthly.b5001.current.ascii',\
    seasonmonths=[9,10,11,12,1,2,3,4,5]):
    '''Average NAO index over a given season'''
    arr=np.loadtxt(filename)
    year, mn, naoi = arr[:,0], arr[:,1], arr[:,2]
    yearmn = [int('%d%02d' %(year[i],mn[i])) for i in xrange(len(year))]
    yearmn = np.asarray(yearmn)
    istart=np.where(mn==seasonmonths[0])[0][0]
    iend=np.where(mn==seasonmonths[-1])[0][-1]
    years=np.unique(year[istart:iend])[:-1]
    naoisel=np.zeros((len(years),len(seasonmonths)))
    for cnt,y in enumerate(years):
        i1 = np.where( yearmn==int('%d%02d' %(y,seasonmonths[0])) )[0][0]
        i2 = np.where( yearmn==int('%d%02d' %(y+1,seasonmonths[-1])) )[0][0]
        naoisel[cnt,:] = naoi[i1:i2+1]
    return years, naoisel

def csiblobprop2attribute(trk,attrb):
    ''' trk = csiblobprop2attribute(trk,attrb)'''
    iattrb=trk.csiblob_varstrs.index(attrb)
    alist=[]
    for i in xrange(trk.nt):
        alist.append(trk.csiblob_stats[i][iattrb])
    exec('trk.'+attrb+'=np.asarray(alist)')
    return trk

def sjpots2attribute(trk):
    englist=[]
    for ky in trk.sjpots.keys():
        potsd = trk.sjpots[ky]
        energy = potsd['energy'][0]
        englist.append(len(energy))
        #englist.append(energy.sum())
    trk.dscapevol=englist
    return trk
         


def seasonalcycles(trackdict,seasonmonths=[9,10,11,12,1,2,3,4,5]):
    '''Calculate the monthly and seasonal totals of storms in trackdict,
       using time of max. vorticity as the date stamp'''
    tkeys=trackdict.keys()
    tkeys.sort()
    trkyears=[]
    trkmonths=[]
    trkyrmonths=[]
    for tk in tkeys:
        trk = trackdict[tk]
        refdate = trk.tdates[0][trk.imxv][0]
        yr,mn = int(refdate[:4]), int(refdate[4:6])
        yrmonth = int(refdate[:6])
        trkyears.append(yr)
        trkmonths.append(mn)
        trkyrmonths.append(yrmonth)
    trkyears = np.asarray(trkyears)
    trkmonths = np.asarray(trkmonths)
    trkyrmonths = np.asarray(trkyrmonths)
    years=np.arange(np.unique(trkyears)[0],np.unique(trkyears)[-1])
    nyears=len(years)
    montharr = np.zeros((nyears,len(seasonmonths)))
    montharr[:] = np.nan
    stmonth = "%02d" % (seasonmonths[0])
    endmonth = "%02d" % (seasonmonths[-1])
    cnt=0
    for yr in years:
        m1, m2 = int(str(yr)+stmonth), int(str(yr+1)+endmonth)
        inseason = (trkyrmonths>=m1) & (trkyrmonths<=m2)
        iseason = np.where(inseason)[0]
        if len(iseason)==0:
            cnt+=1
            continue
        seasoncount=[]
        for smn in seasonmonths:
            mncount = (trkmonths[iseason]==smn).sum()
            seasoncount.append(mncount)
        montharr[cnt,:] = seasoncount
        cnt+=1

    return montharr,seasonmonths, montharr.sum(axis=1), years

def matchknownstorms(alltrackdict, stormsdict='predefined'):
    '''Give trklist and knownstorms dictionary with entries as such
       {'stormname':(date,centrelon,centrelat)}
       RETURNS: knownstorms dictionary with entries modified to include
                matched trackid
                {'stormname':(date,centrelon,centrelat,trkid)}'''
    ###DATES OF KNOW SJ STORMS
    if stormsdict=='predefined':
        knownstorms={'Great': (19871015,-5.,50.),\
                     'ERICA4': (19890105,-55.,46.),\
                     'Oratia': (20001029,0.,53.),\
                     'Anna': (20020226,5.,55.),\
                     'Gudrun': (20050107,-10.,55) ,\
                     'Robert': (20111227,-20.,50.),\
                     'Friedhelm': (20111208,-5.,57.),\
                     'Jeannette': (20021027,-10.,52.),\
                     'Ulli': (20120103,-10.,55.),\
                     'Tilo': ( 20071108,-12.,62.),\
                     'SS2013': ( 20051208,-40.,50.)}
    else: knownstorms=stormsdict
    trklist = alltrackdict.values()
    trk_dt=[(trk.trkno,int(trk.tdates[trk.imxv][:-2])) for trk in trklist]
    dtconvert=mdt.strpdate2num('%Y%m%d')
    trknumdays = np.asarray([dtconvert(str(trackd)) for thr,trackd in trk_dt])
    trk_dt = np.asarray(trk_dt)
    newknownstorms={}
    for ks in knownstorms.keys():
        kst = knownstorms[ks]
        dt,sln,slt = kst
        daydiff = trknumdays-dtconvert(str(dt))
        idtx = np.where(np.abs(daydiff) <= 2)[0]
        #print len(idtx)
        closest=500.; trkid_match=False
        for trkno in trk_dt[idtx]:
            trk = alltrackdict[trkno[0]]
            tlon,tlat = trk.vlon, trk.vlat
            slnv, sltv = np.tile(sln,(len(tlon),1)), np.tile(slt,(len(tlon),1))
            distancev = spheredistance(slnv,sltv,tlon,tlat)
            distance = distancev.min() 
            if distance/1000. < closest:
                trkid_match = trkno[0]
                closest = distance/1000.
        if trkid_match:
            print 'Matched %s (%2.1f km)-SJ+:%s' % \
                (ks, closest, alltrackdict[trkid_match].sjstorm)
            print 'Track: %s' % (trkid_match)
            newknownstorms[ks] = (dt,sln,slt,closest,trkid_match)
            trkcsipts(alltrackdict[trkid_match])
        else: 
            print "No match for",ks
    return newknownstorms

def trkcsipts(trk,blob_azimuth_thresh=(-60,100)):
    az1, az2 = blob_azimuth_thresh
    if not hasattr(trk,'sjpots'):return
    sjpk=trk.sjpots.keys()
    sjpk.sort()
    #print len(sjpk)
    for ky in sjpk:
        pstar = trk.sjpots[ky]['pstar'][0]
        alpha = trk.sjpots[ky]['alpha'][0]
        chi = trk.sjpots[ky]['chi'][0]
        r = trk.sjpots[ky]['r'][0]
        energy = trk.sjpots[ky]['energy'][0]
        wpoint = np.ones(trk.sjpots[ky]['pstar'].shape)
        wlen = np.sum(wpoint)
        blob_azimuth = alpha - chi + 180
        if np.all((blob_azimuth > az1) & (blob_azimuth < az2)) and \
                np.all(r>=250): # Is ix-th blob in WCB?
            print "%d invalid CSI pts" %(wlen)
            continue
        else:
            print "%d valid CSI pts" %(wlen)
        
    
def plotseasonbox(scycle,lc='r',alpha=1.,offset=0.):
    bp = dict(linewidth=2,color=lc,alpha=alpha)
    fp = dict(markersize=12,linewidth=1.5,color=lc,alpha=alpha)
    mp = dict(linewidth=2,color=lc,alpha=alpha)
    #wp = dict(linewidth=1.5,color=lc,alpha=alpha)
    wp = dict(linewidth=1.5,color=lc)
    cp = dict(linewidth=1.5,color=lc,alpha=alpha)
    monthstr=['Sept','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May']
    #plt.boxplot(scycle, notch=0, sym='+', vert=1, whis=1.5,boxprops=bp,\
    #flierprops=fp,medianprops=mp,whiskerprops=wp,capprops=cp)
    plt.boxplot(scycle, positions=np.arange(1,scycle.shape[1]+1)+offset, \
                    vert=1, \
                    showfliers=False, showcaps=True, boxprops=bp, \
                    medianprops=mp, whiskerprops=wp, capprops=cp)
    plt.plot(np.arange(1,10), sps.nanmean(scycle,axis=0), lc+'-', lw=3, \
                 alpha=alpha)
    plt.xticks(np.arange(1,10), monthstr)

def plotstormtrack(m,trk,commongrid=False,color='k',markersize=14.,\
                   plotwhat=['all']):
    '''Plot the track of a storm with various detail included
    USAGE: plotstormtrack(m,trk) where m is an initialized basemap object
           and storm is a SjtrackStats.CycloneTrack,
           plotwhat={['all']} is a list specifying what to plot. Includes
           track, mslpmin, vortmax, uvmaxc, uvmaxw'''
    mx,my = m(trk.mlon,trk.mlat)
    #m.plot(mx,my,'k',linewidth=2.)
    vx,vy = m(trk.vlon,trk.vlat)
    if ('all' in plotwhat) or ('track' in plotwhat):
        trkline=m.plot(vx,vy,color=color,linewidth=1.5,c=color,ms=markersize)
    mslpmin, imin = trk.mslp.min(), trk.mslp.argmin()
    if ('all' in plotwhat) or ('mslpmin' in plotwhat):
        mslpline=m.plot(mx[imin],my[imin],marker=r'$\bigoplus$',mec=color,\
                       mfc=color,ms=markersize)
    vortmax, imax = trk.vort.max(), trk.vort.argmax()
    if ('all' in plotwhat) or ('vortmax' in plotwhat):
        vortline=m.plot(vx[imax],vy[imax],marker='$\zeta$',mec=color,mfc=color,\
               ms=markersize)
    
    if commongrid:
        uv = trk.uvfootprint
        maxuvixs = np.unravel_index(uv.argmax(),uv.shape)
        uvx,uvy = m(commongrid[0][maxuvixs], commongrid[1][maxuvixs])
        uv = trk.uvfootprintc
        maxuvixs = np.unravel_index(uv.argmax(),uv.shape)
        uvcx,uvcy = m(commongrid[0][maxuvixs], commongrid[1][maxuvixs])
        if ('all' in plotwhat) or ('uvmaxc' in plotwhat):
            uvcline=m.plot(uvcx,uvcy,marker='$\Rightarrow$',mec=color,\
                          mfc=color,ms=markersize)
        uv = trk.uvfootprintw
        maxuvixs = np.unravel_index(uv.argmax(),uv.shape)
        uvwx,uvwy = m(commongrid[0][maxuvixs], commongrid[1][maxuvixs])
        if ('all' in plotwhat) or ('uvmaxw' in plotwhat):
            uvmline=m.plot(uvwx,uvwy,marker='$\Nearrow$',mec=color,mfc=color,\
                   ms=markersize)
    return trkline
    
def plotknownstorms(m,commongrid,alltrackdict,knownstorms,plotwhat=['all'],\
                    SJonly=True): 
    cols={'Great':'#800000','ERICA4':'#ff0000','Gudrun':'#ff9966',\
           'Friedhelm':'#66ccff','Robert':'#0033cc',\
           'Oratia':'#00cc00','Jeannette':'#99ffcc','SS2013':'#33ccff',\
           'Tilo':'#cc66ff','Ulli':'#6600cc'}
    names=['Great','ERICA4','Oratia','Jeannette','Gudrun','SS2013',\
           'Tilo','Friedhelm','Robert','Ulli']
    linesegments=[] 
    for ks in names:
        trk=alltrackdict[knownstorms[ks][-1]]
        if trk.sjstorm != SJonly:continue
        trkline=plotstormtrack(m,trk,commongrid=commongrid,color=cols[ks],\
                                   plotwhat=plotwhat)
        trkline[0].set_label(ks)
        linesegments.append(trkline[0])
    return (linesegments,names)
        
   

 
