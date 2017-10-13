import numpy as np
cimport numpy as np
cimport cython
#
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ITYPE = np.int32;
ctypedef np.int32_t ITYPE_t

#=====================================================
#  This script contains cython codes for fast budgat anaysis
#  for CESM
#  See Song et al., 2017
#=====================================================

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def pblidx(np.ndarray[DTYPE_t, ndim=2] hbl, np.ndarray[DTYPE_t, ndim=1] zb,\
           np.ndarray[DTYPE_t, ndim=1] zt):
    """
    [ Function ]
    ihb, iht, alpb, alpt = pblidx(hbl, zb, zt)
    
    [ Description ]
    Compute the vertical grid index at the bottom of planetary boundary layer

    [ Input ]
    hbl : "HBLT" in CESM, Boundary-Layer Depth (cm), float32, [ly, lx]
    zb  : "z_w_bot" in CESM, depth from surface to bottom of layer (cm), float32
    zt  : "z_t" in CESM, depth from surface to midpoint of layer (cm), float32

    [ Output ]
    ihb  : zb[ihb[iy,ix]] is the bottom depth of the cell below hbl
    iht  : zt[iht[iy,ix]] is the first center of the cell below hbl
    alpb : alpb = (hbl - zb[ihb-1]) / (zb[ihb] - zb[ihb-1])
    alpt : alpt = (hbl - zt[iht-1]) / (zt[iht] - zt[iht-1])
    """
    cdef int ly = hbl.shape[0]
    cdef int lx = hbl.shape[1]
    cdef int ix, iy, k, val
    cdef np.ndarray[ITYPE_t, ndim=2] ihb = np.zeros([ly, lx], dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=2] iht = np.zeros([ly, lx], dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] alpb = np.zeros([ly, lx], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] alpt = np.zeros([ly, lx], dtype=DTYPE)
    for iy in xrange(ly):
        for ix in xrange(lx):
            if (hbl[iy,ix] > 0):
                # zb[ihb[iy,ix]] is the bottom depth of the cell
                for k in xrange(len(zb)):
                    if (hbl[iy,ix]-zb[k]) < 0:
                        break
                val = k*1
                #val=[k for k in range(len(zb)) if (hbl[iy,ix]-zb[k])<0][0]
                # zb[ihb[iy,ix]] is the bottom depth of the cell
                ihb[iy,ix] = val
                # alp = (hbl-zb)/dz
                if val == 0:  # if the mld locates at the first layer
                    alpb[iy,ix] = (hbl[iy,ix])/(zb[val])
                else:
                    alpb[iy,ix] = (hbl[iy,ix]-zb[val-1])/(zb[val]-zb[val-1])
       
                # zt[iht[iy,ix]] is the first center of the cell below hbl
                for k in xrange(len(zt)):
                    if (hbl[iy,ix]-zt[k]) < 0:
                        break
                val = k*1
                #val=[k for k in range(len(zt)) if (hbl[iy,ix]-zt[k])<0][0]
                iht[iy,ix] = val
                # alp = (hbl-zt)/dz
                if val == 0:  # if the mld locates at the first layer
                    alpt[iy,ix] = (hbl[iy,ix]-zt[0])/(zt[val]-zt[0])
                else:
                    alpt[iy,ix] = (hbl[iy,ix]-zt[val-1])/(zt[val]-zt[val-1])

    return ihb, iht, alpb, alpt

#@cython.boundscheck(False) # turn of bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getthick(np.ndarray[ITYPE_t, ndim=2] kmt, np.ndarray[DTYPE_t, ndim=2] ht,\
             np.ndarray[DTYPE_t, ndim=1] dz):
    """
    [ Function ]
    thick = getthick(kmt, ht, dz)

    [ Description ]
    Compute the thickness of each grid by taking into account the bottom of the ocean.

    [ Input ]
    kmt : "KMT" in CESM, k Index of Deepest Grid Cell on T Grid, float32, [ly, lx]
    ht  : "HT" in CESM, ocean depth at T points (cm), float32, [ly, lx]
    dz  : "dz" in CESM, thickness of layer k (cm), float32

    [ Output ]
    thick : thickness of each grid cell (cm), float32, [lz, ly, lx]
    """
    cdef int ly = kmt.shape[0]
    cdef int lx = kmt.shape[1]
    cdef int lz = len(dz)
    cdef int ix, iy, iz, cb
    cdef np.ndarray[DTYPE_t, ndim=3] thick = np.ones([lz,ly,lx], dtype=DTYPE)
    cdef DTYPE_t ldep = 0.0

    for ix in xrange(lx):
        for iy in xrange(ly):
            ldep = 0.0
            cb = kmt[iy,ix]-1
            for iz in xrange(cb):
                thick[iz,iy,ix] = dz[iz]
                ldep += dz[iz]
            thick[cb,iy,ix] = ht[iy,ix]-ldep
            for iz in xrange(cb+1,lz):
                thick[iz,iy,ix] = 0.0

    return thick

def mldadv(fname,ihb,alp,thick,tarea,fillval):
    from netCDF4 import Dataset
    """
     Advection over MLD, python style code
    """
    # read netcdf 
    ly=ihb.shape[0]
    lx=ihb.shape[1]
    adv=np.zeros([ly,lx])

    f=Dataset(fname)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if thick[0,iy,ix]==fillval:
                adv[iy,ix]=fillval;
            else:
                #
                # First, vertically integrate hadv from the surface
                # to the top of the cell where the MLD lies.
                # At the cell where MLD lies, some fraction of hadv and vadv
                # is included.
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                for iz in xrange(0,intk+1):
                    # ADVx[ix] = UET[ix-1]-UET[ix]
                    tmp=f.variables['UE_Fe'][0,iz,iy,ix];       # mmol/m^3/s
                    if ix==0:
                        tmp_m1=f.variables['UE_Fe'][0,iz,iy,lx-1];
                        thick_m1=thick[iz,iy,lx-1];
                        tarea_m1=tarea[iy,lx-1];
                    else:
                        tmp_m1=f.variables['UE_Fe'][0,iz,iy,ix-1];
                        thick_m1=thick[iz,iy,ix-1];
                        tarea_m1=tarea[iy,ix-1];
                    ADVx=(tmp_m1*tarea_m1*thick_m1\
                         -tmp*tarea[iy,ix]*thick[iz,iy,ix])*1e-6;
    
                    # ADVy[iy] = VNT[iy-1]-VNT[iy]
                    tmp=f.variables['VN_Fe'][0,iz,iy,ix];
                    if iy==0:
                        tmp_m1=f.variables['VN_Fe'][0,iz,ly-1,ix];
                        thick_m1=thick[iz,ly-1,ix];
                        tarea_m1=tarea[ly-1,ix];
                    else:
                        tmp_m1=f.variables['VN_Fe'][0,iz,iy-1,ix];
                        thick_m1=thick[iz,iy-1,ix];
                        tarea_m1=tarea[iy-1,ix];
                    ADVy=(tmp_m1*tarea_m1*thick_m1\
                         -tmp*tarea[iy,ix]*thick[iz,iy,ix])*1e-6;
    
                    if iz < intk:
                        adv[iy,ix]+=(ADVx+ADVy);           # mmol/s
                    elif iz == intk:
                        adv[iy,ix]+=(ADVx+ADVy)*alp[iy,ix];
                        # consider vertical advection
                        # ADVz[iz] = WT[iz+1]-WT[iz]
                        tmp=f.variables['WT_Fe'][0,iz,iy,ix]*thick[iz,iy,ix];
                        if iz == 61:
                            tmp1=tmp*0.0
                        else:
                            tmp1=f.variables['WT_Fe'][0,iz+1,iy,ix]*thick[iz+1,iy,ix]
                        # linear interpolation between top and bottom of the cell
                        ADVz=(tmp1*alp[iy,ix]+tmp*(1-alp[iy,ix]))*tarea[iy,ix]*1e-6;
                        adv[iy,ix]+=ADVz;
    f.close()
    return adv

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mldadvx(np.ndarray[DTYPE_t, ndim=3] uet, np.ndarray[DTYPE_t, ndim=3] thick,\
            np.ndarray[ITYPE_t, ndim=2] ihb, np.ndarray[DTYPE_t, ndim=2] alp,\
            np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ Function ] 
    vout = mldadvx(uet, thick, ihb, alp, tarea, fillval)

    [ Description ]
    Vertically integrated advection over MLD in x direction

    [ Input ]
    uet     : Tracer flux in x direction (tracer's unit per s), 
              float32, [lz, ly, lx]
    thick   : Thinkness of each model grid (cm), float32, [lz, ly, lx]
    ihb     : zb[ihb] is the bottom depth of the cell, int32, [ly, lx]
    alp     : alp = (hbl - zb[ihb-1]) / (zb[ihb] - zb[ihb-1]), float32, [ly, lx]
    tarea   : "TAREA" in CESM, area of T cells (cm^2), float32, [ly, lx]
    fillval : fillvalue for a tracer. It is in the CESM output file

    [ Output ]
    vout : vertically integrated advection over MLD in x direction
           (tracer's unit per s), float32, [ly, lx]
    """
    cdef int ly = uet.shape[1]
    cdef int lx = uet.shape[2]
    cdef int ix, iy, iz, intk
    cdef DTYPE_t tmp, tmp_m1, ADVx
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if thick[0,iy,ix] == fillval:
                vout[iy,ix] = fillval
            else:
                #
                # First, vertically integrate hadv from the surface
                # to the top of the cell where the MLD lies.
                # At the cell where MLD lies, some fraction of hadv and vadv
                # is included.
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                for iz in xrange(intk+1):
                    # ADVx[ix] = UET[ix-1]-UET[ix]
                    tmp=uet[iz,iy,ix];       # mmol/m^3/s
                    if ix==0:
                        tmp_m1=uet[iz,iy,lx-1];
                        thick_m1=thick[iz,iy,lx-1];
                        tarea_m1=tarea[iy,lx-1];
                    else:
                        tmp_m1=uet[iz,iy,ix-1];
                        thick_m1=thick[iz,iy,ix-1];
                        tarea_m1=tarea[iy,ix-1];
                    ADVx=(tmp_m1*tarea_m1*thick_m1\
                         -tmp*tarea[iy,ix]*thick[iz,iy,ix])*1e-6;
    
                    if iz < intk:
                        vout[iy,ix] += ADVx           # mmol/s
                    elif iz == intk:
                        vout[iy,ix] += ADVx*alp[iy,ix]
    return vout 

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mldadvy(np.ndarray[DTYPE_t, ndim=3] vnt, np.ndarray[DTYPE_t, ndim=3] thick,\
            np.ndarray[ITYPE_t, ndim=2] ihb, np.ndarray[DTYPE_t, ndim=2] alp,\
            np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ Function ]
    vout = mldadvy(vnt, thick, ihb, alp, tarea, fillval)

    [ Description ]
    Vertically integrated advection over MLD in y direction

    [ Input ]
    vnt     : Tracer flux in y direction (tracer's unit per s),
              float32, [lz, ly, lx]
    thick   : Thinkness of each model grid (cm), float32, [lz, ly, lx]
    ihb     : zb[ihb] is the bottom depth of the cell, int32, [ly, lx]
    alp     : alp = (hbl - zb[ihb-1]) / (zb[ihb] - zb[ihb-1]), float32, [ly, lx]
    tarea   : "TAREA" in CESM, area of T cells (cm^2), float32, [ly, lx]
    fillval : fillvalue for a tracer. It is in the CESM output file

    [ Output ]
    vout : vertically integrated advection over MLD in y direction
           (tracer's unit per s), float32, [ly, lx]
    """
    cdef int ly = vnt.shape[1]
    cdef int lx = vnt.shape[2]
    cdef int ix, iy, iz, intk
    cdef DTYPE_t tmp, tmp_m1, thick_m1, tarea_m1, ADVy
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if thick[0,iy,ix] == fillval:
                vout[iy,ix] = fillval
            else:
                #
                # First, vertically integrate hadv from the surface
                # to the top of the cell where the MLD lies.
                # At the cell where MLD lies, some fraction of hadv and vadv
                # is included.
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                for iz in xrange(intk+1):
                    # ADVy[ix] = VNT[ix-1]-VNT[ix]
                    tmp=vnt[iz,iy,ix];       # mmol/m^3/s
                    if iy==0:
                        tmp_m1=vnt[iz,ly-1,ix];
                        thick_m1=thick[iz,ly-1,ix];
                        tarea_m1=tarea[ly-1,ix];
                    else:
                        tmp_m1=vnt[iz,iy-1,ix];
                        thick_m1=thick[iz,iy-1,ix];
                        tarea_m1=tarea[iy-1,ix];
                    ADVy=(tmp_m1*tarea_m1*thick_m1\
                         -tmp*tarea[iy,ix]*thick[iz,iy,ix])*1e-6;

                    if iz < intk:
                        vout[iy,ix] += ADVy           # mmol/s
                    elif iz == intk:
                        vout[iy,ix] += ADVy*alp[iy,ix]
    return vout

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mldadvz(np.ndarray[DTYPE_t, ndim=3] wtt, np.ndarray[DTYPE_t, ndim=3] thick,\
            np.ndarray[ITYPE_t, ndim=2] ihb, np.ndarray[DTYPE_t, ndim=2] alp,\
            np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ Function ]
    vout = mldadvz(wtt, thick, ihb, alp, tarea, fillval)

    [ Description ]
    Vertically integrated advection over MLD in z direction

    [ Input ]
    wtt     : Tracer flux in z direction (tracer's unit per s),
              float32, [lz, ly, lx]
    thick   : Thinkness of each model grid (cm), float32, [lz, ly, lx]
    ihb     : zb[ihb] is the bottom depth of the cell, int32, [ly, lx]
    alp     : alp = (hbl - zb[ihb-1]) / (zb[ihb] - zb[ihb-1]), float32, [ly, lx]
    tarea   : "TAREA" in CESM, area of T cells (cm^2), float32, [ly, lx]
    fillval : fillvalue for a tracer. It is in the CESM output file

    [ Output ]
    vout : vertically integrated advection over MLD in z direction
           (tracer's unit per s), float32, [ly, lx]
    """
    cdef int ly = wtt.shape[1]
    cdef int lx = wtt.shape[2]
    cdef int ix, iy, iz, intk
    cdef DTYPE_t bot, top, ADVz
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if thick[0,iy,ix] == fillval:
                vout[iy,ix] = fillval
            else:
                #
                # First, vertically integrate vadv from the surface
                # to the top of the cell where the MLD lies.
                # At the cell where MLD lies, estimate vertical flux 
                # across the MLD using alpha.
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                # ADVz[iz] = WT[iz+1]-WT[iz]
                top=wtt[intk,iy,ix]*thick[intk,iy,ix];
                if intk == 61:
                    bot=top*0.0
                else:
                    bot=wtt[intk+1,iy,ix]*thick[intk+1,iy,ix]
                # linear interpolation between top and bottom of the cell
                ADVz=(bot*alp[iy,ix]+top*(1-alp[iy,ix]))*tarea[iy,ix]*1e-6;
                vout[iy,ix]=ADVz;
                #vout[iy,ix]+=ADVz;
    return vout

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mlddifx(np.ndarray[DTYPE_t, ndim=3] dfx, np.ndarray[DTYPE_t, ndim=3] thick,\
            np.ndarray[ITYPE_t, ndim=2] ihb, np.ndarray[DTYPE_t, ndim=2] alp,\
            np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ Function ]
    vout = mlddifx(dfx, thick, ihb, alp, tarea, fillval)

    [ Description ]
    Vertically integrated diffusion over MLD in x direction

    [ Input ]
    dfx     : Tracer diffusive flux in x direction (tracer's unit per s),
              float32, [lz, ly, lx]
    thick   : Thinkness of each model grid (cm), float32, [lz, ly, lx]
    ihb     : zb[ihb] is the bottom depth of the cell, int32, [ly, lx]
    alp     : alp = (hbl - zb[ihb-1]) / (zb[ihb] - zb[ihb-1]), float32, [ly, lx]
    tarea   : "TAREA" in CESM, area of T cells (cm^2), float32, [ly, lx]
    fillval : fillvalue for a tracer. It is in the CESM output file

    [ Output ]
    vout : vertically integrated diffusion over MLD in x direction
           (tracer's unit per s), float32, [ly, lx]
    """
    cdef int ly = dfx.shape[1]
    cdef int lx = dfx.shape[2]
    cdef int ix, iy, iz, intk
    cdef DTYPE_t tmp, tmp_m1, thick_m1, tarea_m1, DFFx
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if thick[0,iy,ix] == fillval:
                vout[iy,ix] = fillval
            else:
                #
                # First, vertically integrate hdif from the surface
                # to the top of the cell where the MLD lies.
                # At the cell where MLD lies, some fraction of hdif is included.
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                for iz in xrange(intk+1):
                    # DFFx[ix] = -HDIFE[ix-1]-(-HDIFE[ix])
                    tmp=-dfx[iz,iy,ix];       # mmol/m^3/s
                    if ix==0:
                        tmp_m1=-dfx[iz,iy,lx-1];
                        thick_m1=thick[iz,iy,lx-1];
                        tarea_m1=tarea[iy,lx-1];
                    else:
                        tmp_m1=-dfx[iz,iy,ix-1];
                        thick_m1=thick[iz,iy,ix-1];
                        tarea_m1=tarea[iy,ix-1];
                    DFFx=(tmp_m1*tarea_m1*thick_m1\
                         -tmp*tarea[iy,ix]*thick[iz,iy,ix])*1e-6;

                    if iz < intk:
                        vout[iy,ix] += DFFx           # mmol/s
                    elif iz == intk:
                        vout[iy,ix] += DFFx*alp[iy,ix]
    return vout

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mlddify(np.ndarray[DTYPE_t, ndim=3] dfy, np.ndarray[DTYPE_t, ndim=3] thick,\
            np.ndarray[ITYPE_t, ndim=2] ihb, np.ndarray[DTYPE_t, ndim=2] alp,\
            np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ Function ]
    vout = mldadvy(dfy, thick, ihb, alp, tarea, fillval)

    [ Description ]
    Vertically integrated diffusion over MLD in y direction

    [ Input ]
    dfy     : Tracer diffusive flux in y direction (tracer's unit per s),
              float32, [lz, ly, lx]
    thick   : Thinkness of each model grid (cm), float32, [lz, ly, lx]
    ihb     : zb[ihb] is the bottom depth of the cell, int32, [ly, lx]
    alp     : alp = (hbl - zb[ihb-1]) / (zb[ihb] - zb[ihb-1]), float32, [ly, lx]
    tarea   : "TAREA" in CESM, area of T cells (cm^2), float32, [ly, lx]
    fillval : fillvalue for a tracer. It is in the CESM output file

    [ Output ]
    vout : vertically integrated diffusion over MLD in y direction
           (tracer's unit per s), float32, [ly, lx]
    """
    cdef int ly = dfy.shape[1]
    cdef int lx = dfy.shape[2]
    cdef int ix, iy, iz, intk
    cdef DTYPE_t tmp, tmp_m1, thick_m1, tarea_m1, DFFy
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if thick[0,iy,ix] == fillval:
                vout[iy,ix] = fillval
            else:
                #
                # First, vertically integrate hdif from the surface
                # to the top of the cell where the MLD lies.
                # At the cell where MLD lies, some fraction of hdif is included.
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                for iz in xrange(0,intk+1):
                    # DFFy[ix] = -HDIFN[ix-1]-(-HDIFN[ix])
                    tmp=-dfy[iz,iy,ix];       # mmol/m^3/s
                    if iy==0:
                        tmp_m1=-dfy[iz,ly-1,ix];
                        thick_m1=thick[iz,ly-1,ix];
                        tarea_m1=tarea[ly-1,ix];
                    else:
                        tmp_m1=-dfy[iz,iy-1,ix];
                        thick_m1=thick[iz,iy-1,ix];
                        tarea_m1=tarea[iy-1,ix];
                    DFFy=(tmp_m1*tarea_m1*thick_m1\
                         -tmp*tarea[iy,ix]*thick[iz,iy,ix])*1e-6;

                    if iz < intk:
                        vout[iy,ix] += DFFy           # mmol/s
                    elif iz == intk:
                        vout[iy,ix] += DFFy*alp[iy,ix]
    return vout 


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mlddiam(np.ndarray[DTYPE_t, ndim=3] var,\
            np.ndarray[ITYPE_t, ndim=2] ihb, np.ndarray[DTYPE_t, ndim=2] alp,\
            np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ Function ]
    dia = mlddiam(var, ihb, alp, tarea, fillval)

    [ Description ]
    Vertically integrated diapycnal mixing in the MLD

    [ Input ]
    var     : Tracer diapycnal vertical mixing (tracer's unit* cm/s),
              float32, [lz, ly, lx]
    ihb     : zb[ihb] is the bottom depth of the cell, int32, [ly, lx]
    alp     : alp = (hbl - zb[ihb-1]) / (zb[ihb] - zb[ihb-1]), float32, [ly, lx]
    tarea   : "TAREA" in CESM, area of T cells (cm^2), float32, [ly, lx]
    fillval : fillvalue for a tracer. It is in the CESM output file

    [ Output ]
    vout : vertically integrated diffusion over MLD in y direction
           (tracer's unit per s), float32, [ly, lx]
    """
    cdef int ly = var.shape[1]
    cdef int lx = var.shape[2]
    cdef int ix, iy, iz, intk
    cdef DTYPE_t bot, top
    cdef np.ndarray[DTYPE_t, ndim=2] dia = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if var[0,iy,ix] == fillval:
                dia[iy,ix] = fillval
            else:
                #
                # At the cell where MLD lies, perform linear interpolation
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                # DIA is defined at z_w_bottom, mmol/m^3 cm/s
                # and multiply (-) to convert upward
                bot = -var[intk,iy,ix]
                if intk == 0:
                    top = bot*0.0
                else:
                    top = -var[intk-1,iy,ix]
                # linear interpolation between top and bottom of the cell
                dia[iy,ix] = (bot*alp[iy,ix]+top*(1-alp[iy,ix]))*tarea[iy,ix]*1e-6
                #dia[iy,ix]+=DIAM;
    return dia

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mldvint(np.ndarray[DTYPE_t, ndim=3] var, np.ndarray[DTYPE_t, ndim=3] thick,\
            np.ndarray[ITYPE_t, ndim=2] ihb, np.ndarray[DTYPE_t, ndim=2] alp,\
            np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    Vetical integration over the PBL
    """
    cdef int ly = var.shape[1]
    cdef int lx = var.shape[2]
    cdef int ix, iy, iz, intk
    cdef DTYPE_t tmp, vint
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if thick[0,iy,ix]==fillval:
                vout[iy,ix]=fillval;
            else:
                #
                # First, vertically integrate hdff from the surface
                # to the top of the cell where the MLD lies.
                # At the cell where MLD lies, some fraction of hdff and vdff
                # is included.
                #
                # zb[ihb] is the bottom depth and zt[ihb] is the center depth
                intk = ihb[iy,ix]
                for iz in xrange(0,intk+1):
                    tmp=var[iz,iy,ix];       # mmol/m^3
                    vint=tmp*tarea[iy,ix]*thick[iz,iy,ix]*1e-6;
                    if iz < intk:
                        vout[iy,ix]+=vint;           # mmol
                    elif iz == intk:
                        vout[iy,ix]+=vint*alp[iy,ix];
    return vout

#=========================================================
#  Entrainment / detrainment
#  consider the change in PBL between t_i and t_(i+1) 
#  and multiply Fe(h(t_1))
#=========================================================

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def entbt(np.ndarray[DTYPE_t, ndim=3] trc, np.ndarray[DTYPE_t, ndim=2] dhbl,\
          np.ndarray[ITYPE_t, ndim=2] iht, np.ndarray[DTYPE_t, ndim=2] alp,\
          np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ function ]
    vout = entbt(fe, dhbl, iht, tarea, fillval)
    
    [ description ]
    compute tracer(t,h(t))*dh(t)/dt. Using the forward difference to compute dh(t)/dt

    [ input ]
    trc     : Tracer (for Fe, mmol/m3), float32, [lz, ly, lx]
    dhbl    : (hbl(t+delt) - hbl(t))/dt, 2D array, float32, (cm/s)
    iht     : vertical index where z_t[iht] is the depth of the first cell
              below the hbl, int32, [ly, lx]
    alp     : weight for the linear interpolation of Fe(hbl), float32, [ly, lx]
    tarea   : area of each grid cell (cm^2), float32, [ly, lx]
    fillval : the value for the masked element in fe, float32

    [ output ]
    vout : entrainment at the base of the MLD (mmol/s), float32, [ly, lx]
    """
    cdef int ly = trc.shape[1]
    cdef int lx = trc.shape[2]
    cdef int ix, iy, ik
    cdef DTYPE_t tmp
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if trc[0,iy,ix]==fillval:
                vout[iy,ix]=fillval;
            else:
                # Fe at the PBL
                # z_t[iht] is the ceter depth of the first cell below PBL
                ik = iht[iy,ix];
                if trc[ik,iy,ix]==fillval:
                    bot=trc[ik-1,iy,ix];
                    #bot=0.0
                else:
                    bot=trc[ik,iy,ix];
                top=trc[ik-1,iy,ix]
                tmp=(bot*alp[iy,ix]+top*(1-alp[iy,ix]));
                #
                # entrainment 
                vout[iy,ix] = tmp*dhbl[iy,ix]*tarea[iy,ix]*1e-6;
                
    return vout

def enttp(np.ndarray[DTYPE_t, ndim=2] trc, np.ndarray[DTYPE_t, ndim=2] dssh,\
          np.ndarray[DTYPE_t, ndim=2] tarea, DTYPE_t fillval):
    """
    [ function ]
    vout = entbt(fe, dhbl, iht, tarea, fillval)

    [ description ]
    compute TR(t)*dh(t)/dt. Using the forward difference to compute dh(t)/dt

    [ input ]
    trc     : Tracer, 3D array, float32, (for Fe, mmol/m3)
    dssh    : (ssh(t+delt) - ssh(t))/dt, 2D array, float32, (cm/s)
    iht     : vertical index where z_t[iht] is the depth of the first cell
              below the hbl, 2D array, int
    alp     : weight for the linear interpolation for Fe(hbl), 2D array
    tarea   : area of each grid cell, 2D array, (cm^2)
    fillval : the value for the masked element in fe, float32

    [ output ]
    vout : entrainment term, 2D array, (mmol/s)
    """
    cdef int ly = trc.shape[0]
    cdef int lx = trc.shape[1]
    cdef int ix, iy, ik
    cdef DTYPE_t tmp
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)
    #
    for ix in xrange(lx):
        for iy in xrange(ly):
            if trc[iy,ix]==fillval:
                vout[iy,ix]=fillval;
            else:
                # Fe at the PBL
                # z_t[iht] is the ceter depth of the first cell below PBL
                # entrainment
                vout[iy,ix] = trc[iy,ix]*dssh[iy,ix]*tarea[iy,ix]*1e-6;

    return vout

#=========================================================
#  Compute anomalies w.r.t. spatial mean
#=========================================================

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) 
def computeanom(np.ndarray[DTYPE_t, ndim=2] varin, np.ndarray[DTYPE_t, ndim=2] dxt,\
                np.ndarray[DTYPE_t, ndim=2] dyt, np.ndarray[DTYPE_t, ndim=1] lscale,\
                ITYPE_t ybuf, DTYPE_t fillval):
    """
    [ function ]
    vout = computeanom(varin, dxt, dyt, lscale, ybuf, fillval)

    [ description ]
    Compute the anomaly with respect to the spatial mean.

    [ input ]
    varin   : input variable, 2D array
    dxt     : distance in x direction, 2D array (km)
    dyt     : distance in y direction, 2D array (km)
    lscale  : radius scale Ls from Fig. 12 in Chelton et al.(2011), 1D array, (km)
    ybuf    : number of grid points for buffering zone in y direction, constant
    fillval : the value for the masked element in fe, float32

    [ output ]
    vout    : anomalies wrt the spatial mean, 2D array
    """
    cdef int ii, jj, ix, iy, nx, ny, ju, jl, icnt
    cdef int ly = dxt.shape[0]
    cdef int lx = dxt.shape[1]
    cdef DTYPE_t dx, dy, isum
    cdef np.ndarray[DTYPE_t, ndim=2] vout = np.zeros([ly, lx], dtype=DTYPE)

    for jj in xrange(ly):
        dy = dyt[jj,0]
        ny = int(lscale[jj]*2.0/dy);
        jl = max(0,jj-ny)
        ju = min(ly+ybuf,jj+ny+1);
        for ii in xrange(lx):
            isum=0.0;
            icnt=0;
            if varin[jj,ii]!=fillval:
                dx=dxt[jj,ii];
                nx=int(lscale[jj]*2.0/dx);
                for ix in xrange(ii-nx,ii+nx+1):
                    if ix<0:
                       ix=lx+ix
                    elif ix>(lx-1):
                       ix=ix-lx
                    for iy in xrange(jl,ju):
                        if varin[iy,ix]!=fillval:
                            isum+=varin[iy,ix];
                            icnt+=1
                vout[jj,ii]=varin[jj,ii]-isum/icnt
    return vout
