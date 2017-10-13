import numpy as np
from netCDF4 import Dataset
import glob
import os.path
import tracerbudget as cesmpp
"""
The main script for the budget computation of Fe averaged over the MLD.
It can be applied to other biogeochemical tracers.
This script calls functions from cesmpp that is written in Cython.

2016.07.19, Hajoon Song
"""
#
#=========================================================
#  User definition
#=========================================================
#
sdir='/glade/scratch/mclong/hi-res-eco/g.e11.G.T62_t12.eco.006/ocn/hist/';
odir='/glade/scratch/hajoon/cesm/'
outname='FeMLD';
ymax=868; 
#ymax=868+20; 
delt=10*86400.0;           # this is 2*delt 
                           # because we use centered descritization
#
#=========================================================
#  Model grid information
#=========================================================
#
f=Dataset(sdir+'g.e11.G.T62_t12.eco.006.pop.h.0001-09-02.nc');
# 1/10 degree
dz = f.variables['dz'][:];
zt = f.variables['z_t'][:];
zb = f.variables['z_w_bot'][:];
ht = f.variables['HT'][:ymax,:];
kmt = f.variables['KMT'][:ymax,:];
tarea = f.variables['TAREA'][:ymax,:]
f.close()
ht = ht.astype('float32');
tarea = tarea.astype('float32');
# adjust thickness of the bottom level
thick=cesmpp.getthick(kmt,ht,dz);
[lz,ly,lx]=thick.shape
kmt=None;ht=None;
#
#=========================================================
#  Read model variables and compute the budget terms
#=========================================================
#
files=sorted(glob.glob(sdir+'g.e11.G.T62_t12.eco.006.pop.h.000*.nc'));
#
# Loop over trng and add up terms
# This code estimates the tendency using centered discretization
# We need HBL, SSH and vertically integrated Fe from t-delt, t and t+delt.
#
hbl=np.zeros([3,ly,lx],dtype='float32');
ssh=np.zeros([3,ly,lx],dtype='float32');
vFe=np.zeros([3,ly,lx],dtype='float32');
mld=np.zeros([3,ly,lx],dtype='float32');
# we need data from (trng-delt) to (trng+delt)
for i in xrange(1,6):
#for i in xrange(340,341):
    fname = files[i]
    newfname=outname+'_'+fname[-10]+fname[-8:-6]+fname[-5:-3]+'.npz';
    #newfname=odir+outname+'_'+fname[-10]+fname[-8:-6]+fname[-5:-3]+'.npz';
    print "working at t="+fname[-10]+fname[-8:-6]+fname[-5:-3]
    #
    # We need to load and process HBL, SSH and vertically integrated Fe 
    # from t-delt, t and t+delt.
    #
    # At first, prepare terms at t-delt
    #
    f=Dataset(files[i-1]);
    var=f.variables['HBLT'][0,:ymax,:];
    hbl[0,:,:]=var.filled(0);
    var=f.variables['SSH'][0,:ymax,:];
    ssh[0,:,:]=var.filled();
    thick[0,:,:]=dz[0]+ssh[0,:,:]
    [ihb,iht,alp,alpt]=cesmpp.pblidx(hbl[0,...],zb,zt);
    zmax=ihb.max()+1;
    var=f.variables['Fe'][0,:zmax,:ymax,:];
    fval=var.fill_value;
    var=var.filled();
    vFe[0,:,:]=cesmpp.mldvint(var,thick,ihb,alp,tarea,fval);
    f.close();
    mld[0,:,:]=(ssh[0,:,:]+hbl[0,:,:])*1e-2;  # cm to m
    #
    # Prepare terms at t
    #
    f=Dataset(fname);
    var=f.variables['HBLT'][0,:ymax,:];
    hbl[1,:,:]=var.filled(0);
    var=f.variables['SSH'][0,:ymax,:];
    ssh[1,:,:]=var.filled();
    thick[0,...]=dz[0]+ssh[1,:,:];
    [ihb,iht,alp,alpt]=cesmpp.pblidx(hbl[1,...],zb,zt);
    zmax=ihb.max()+1;
    var=f.variables['Fe'][0,:zmax,:ymax,:];
    fval=var.fill_value;
    var=var.filled();
    vFe[1,:,:]=cesmpp.mldvint(var,thick,ihb,alp,tarea,fval);
    f.close();
    mld[1,:,:]=(ssh[1,:,:]+hbl[1,:,:])*1e-2;   # cm to m
    #
    # Then prepare terms for t+delt
    #
    f=Dataset(files[i+1]);
    var=f.variables['HBLT'][0,:ymax,:];
    hbl[2,:,:]=var.filled(0);
    var=f.variables['SSH'][0,:ymax,:];
    ssh[2,:,:]=var.filled();
    thick[0,:,:]=dz[0]+ssh[2,:,:]
    [ihb,iht,alp,alpt]=cesmpp.pblidx(hbl[2,...],zb,zt);
    zmax=ihb.max()+1;
    var=f.variables['Fe'][0,:zmax,:ymax,:];
    fval=var.fill_value;
    var=var.filled();
    vFe[2,:,:]=cesmpp.mldvint(var,thick,ihb,alp,tarea,fval);
    f.close();
    mld[2,:,:]=(ssh[2,:,:]+hbl[2,:,:])*1e-2; # cm to m
    #
    # compute LHS
    #
    LHS=(vFe[2,:,:]/mld[2,:,:]-vFe[0,:,:]/mld[0,:,:])/delt
    #
    # At t, compute RHS 
    #
    thick[0,...]=dz[0];
    [ihb,iht,alp,alpt]=cesmpp.pblidx(hbl[1,...],zb,zt);
    zmax=ihb.max()+1;
    f=Dataset(fname);
    #
    # 1. advection
    #
    adv=np.zeros([ly,lx],dtype='float32');
    var=f.variables['UE_Fe'][0,:zmax,:ymax,:];
    adv+=cesmpp.mldadvx(var,thick,ihb,alp,tarea,fval);
    var=f.variables['VN_Fe'][0,:zmax,:ymax,:];
    adv+=cesmpp.mldadvy(var,thick,ihb,alp,tarea,fval);
    var=f.variables['WT_Fe'][0,:(zmax+1),:ymax,:];
    adv+=cesmpp.mldadvz(var,thick,ihb,alp,tarea,fval);
    adv=adv/(mld[1,:,:])
    #
    # 2. diffusion
    #
    dif=np.zeros([ly,lx],dtype='float32');
    var=f.variables['HDIFE_Fe'][0,:zmax,:ymax,:];
    dif+=cesmpp.mlddifx(var,thick,ihb,alp,tarea,fval);
    var=f.variables['HDIFN_Fe'][0,:zmax,:ymax,:];
    dif+=cesmpp.mlddify(var,thick,ihb,alp,tarea,fval);
    dif=dif/(mld[1,:,:])
    #
    # 3. diffusion
    #
    var=f.variables['DIA_IMPVF_Fe'][0,:zmax,:ymax,:];
    var=var.filled();
    dia=cesmpp.mlddiam(var,ihb,alp,tarea,fval)/(mld[1,:,:]);
    #
    # 4. KPP nonlocal term
    #
    var=f.variables['KPP_SRC_Fe'][0,:zmax,:ymax,:];
    kpp=cesmpp.mldvint(var,thick,ihb,alp,tarea,fval)/(mld[1,:,:]);
    #
    # 5. Source / Sink
    #
    var=f.variables['J_Fe'][0,:zmax,:ymax,:];
    bio=cesmpp.mldvint(var,thick,ihb,alp,tarea,fval)/(mld[1,:,:]);
    #
    # 6. Iron flux from the surface
    #
    var=f.variables['IRON_FLUX'][0,:ymax,:]*tarea*1e-4; # mmol/s
    flx=var.filled();
    flx=flx/(mld[1,:,:])
    #
    # 7. entrainment / detrainment
    #
    var=f.variables['Fe'][0,:(zmax+1),:ymax,:];
    var=var.filled();
    dhbl=(hbl[2,:,:]-hbl[0,:,:]);
    ent=cesmpp.entbt(var,dhbl/delt,iht,alpt,tarea,fval)/mld[1,:,:];
    var=(ssh[2,:,:]-ssh[0,:,:])/delt + dhbl/delt; #cm/s
    ent+=-(var*1e-2)*vFe[1,:,:]/(mld[1,:,:]**2);
    #dhbl=(hbl[2,:,:]-hbl[0,:,:])/delt
    #ent+=cesmpp.entbt(var,dhbl,iht,alpt,tarea,fval)/MLD[1,:,:];

    #
    # 8. Save the analysis
    #
    np.savez(newfname,LHS=LHS,adv=adv,dif=dif,dia=dia,\
             kpp=kpp,bio=bio,flx=flx,ent=ent);

    RHS=adv+dif+dia+bio+kpp+flx+ent

    diff = LHS - RHS
    diff[np.nonzero(diff<-100)] = np.nan
    print " "
    print "error=" + str(np.nanmean(np.nanmean(np.abs(diff),axis=1),axis=0))
