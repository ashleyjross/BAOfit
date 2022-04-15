import BAOfit as bf
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import numpy.linalg as linalg

import pycorr

rmin = 50
rmax = 150
maxb = 80.
binc = 0

zmin = 0.8
zmax = 1.1
bs = 4

sfog = 3 #fog velocity term, 3 is kind of cannonical
dperp = 4 # about right for z ~0.8
drad = 8 # about right for z ~0.8

Nmock = 1000

def sigreg_c12(al,chill,fac=1.,md='f'):
	#report the confidence region +/-1 for chi2
	#copied from ancient code
	chim = 1000
	
	
	chil = []
	for i in range(0,len(chill)):
		chil.append((chill[i],al[i]))
		if chill[i] < chim:
			chim = chill[i]
			am = al[i]
			im = i
	#chim = min(chil)	
	a1u = 2.
	a1d = 0
	a2u = 2.
	a2d = 0
	oa = 0
	ocd = 0
	s0 = 0
	s1 = 0
	for i in range(im+1,len(chil)):
		chid = chil[i][0] - chim
		if chid > 1. and s0 == 0:
			a1u = (chil[i][1]/abs(chid-1.)+oa/abs(ocd-1.))/(1./abs(chid-1.)+1./abs(ocd-1.))
			s0 = 1
		if chid > 4. and s1 == 0:
			a2u = (chil[i][1]/abs(chid-4.)+oa/abs(ocd-4.))/(1./abs(chid-4.)+1./abs(ocd-4.))
			s1 = 1
		ocd = chid	
		oa = chil[i][1]
	oa = 0
	ocd = 0
	s0 = 0
	s1 = 0
	for i in range(1,im):
		chid = chil[im-i][0] - chim
		if chid > 1. and s0 == 0:
			a1d = (chil[im-i][1]/abs(chid-1.)+oa/abs(ocd-1.))/(1./abs(chid-1.)+1./abs(ocd-1.))
			s0 = 1
		if chid > 4. and s1 == 0:
			a2d = (chil[im-i][1]/abs(chid-4.)+oa/abs(ocd-4.))/(1./abs(chid-4.)+1./abs(ocd-4.))
			s1 = 1
		ocd = chid	
		oa = chil[im-i][1]
	if a1u < a1d:
		a1u = 2.
		a1d = 0
	if a2u < a2d:
		a2u = 2.
		a2d = 0
			
	return am,a1d,a1u,a2d,a2u,chim	


#make BAO template given parameters above, using DESI fiducial cosmology and cosmoprimo P(k) tools
#mun is 0 for pre rec
#sigs is only relevant if mun != 0 and should then be the smoothing scale for reconstructions
#beta is b/f, so should be changed depending on tracer
#sp is the spacing in Mpc/h of the templates that get written out, most of the rest of the code assumes 1
#BAO and nowiggle templates get written out for xi0,xi2,xi4 (2D code reconstructions xi(s,mu) from xi0,xi2,xi4)
bf.mkxifile_3dewig(sp=1.,v='y',pkfile='DESI',mun=0,beta=0.4,sfog=sfog,sigt=dperp,sigr=drad,sigs=15.)

#sys.exit()

#make covariance matrix from EZ mocks
#def get_xi0cov():
znm = str(10*zmin)[:1]+str(10*zmax)[:1]
dirm = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/csaulder/EZmocks/'
fnm = 'EZmock_results_'+znm
result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+'_1.npy')
rebinned = result[:(result.shape[0]//bs)*bs:bs]
ells = 0#(0, 2)
s, xiell = rebinned(ells=ells, return_sep=True)
nbin = len(xiell)
print(nbin)
xiave = np.zeros((nbin))
cov = np.zeros((nbin,nbin))

Ntot = 0
fac = 1.
for i in range(1,Nmock+1):
	nr = '_'+str(i)
	result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+nr+'.npy')
	rebinned = result[:(result.shape[0]//bs)*bs:bs]
	xic = rebinned(ells=ells)#[0]
	xiave += xic
	Ntot += 1.
print( Ntot)        
xiave = xiave/float(Ntot)
for i in range(1,Nmock+1):
	nr = '_'+str(i)
	result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+nr+'.npy')
	rebinned = result[:(result.shape[0]//bs)*bs:bs]
	xic = rebinned(ells=ells)#[0]
	for j in range(0,nbin):
		xij = xic[j]
		for k in range(0,nbin):
			xik = xic[k]
			cov[j][k] += (xij-xiave[j])*(xik-xiave[k])

cov = cov/float(Ntot)                   
	
#return cov

#cov = get_xi0cov()
xistd = []
# covn = np.zeros((len(xiave),len(xiave)))
for i in range(0,len(xiave)):
     xistd.append(np.sqrt(cov[i][i]))
#     for j in range(0,len(xiave)):
#         covn[i][j] = cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])
#plt.errorbar(s,s**2.*xiell,s**2.*np.array(xistd))
#plt.show()
#invcov = linalg.inv(cov)
#plt.imshow(invcov)
#plt.imshow(covn)

#plt.show()
#sys.exit()



rl = []
nbin = 0
for i in range(0,len(s)):
    r = s[i]
    #correct for pairs should have slightly larger average pair distance than the bin center
    #this assumes mid point of bin is being used and pairs come from full 3D volume
    rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) 
    rl.append(rbc) 
    if rbc > rmin and rbc < rmax:
        nbin += 1

rl = np.array(rl)
sbaotemp = str(sfog)+str(dperp)+str(drad)
mod = np.loadtxt('BAOtemplates/xi0DESI0.4'+sbaotemp+'15.00.dat').transpose()[1]
modsm = np.loadtxt('BAOtemplates/xi0smDESI0.4'+sbaotemp+'15.00.dat').transpose()[1]

spa=.001
outdir = os.environ['HOME']+'/DESImockbaofits/'

#do abacus cut

abdir = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/jmena/'
xid = np.loadtxt(abdir+'Xi_0_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[0] #just look at first mock

wo = 'abcutsky0'
                
lik = bf.doxi_isolike(xid,cov,mod,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRG'+wo+str(zmin)+str(zmax)+sbaotemp+str(bs),diro=outdir)
print('minimum chi2 is '+str(min(lik))+' for '+str(nbin-5)+' dof')
liksm = bf.doxi_isolike(xid,cov,modsm,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRG'+wo+'_smooth_'+str(zmin)+str(zmax)+sbaotemp+str(bs),diro=outdir)
#print(lik)
#print(liksm)
al = [] #list to be filled with alpha values
for i in range(0,len(lik)):
    a = .8+spa/2.+spa*i
    al.append(a)
#below assumes you have matplotlib to plot things, if not, save the above info to a file or something

sigs = sigreg_c12(al,lik)
print('result is alpha = '+str((sigs[2]+sigs[1])/2.)+'+/-'+str((sigs[2]-sigs[1])/2.))


plt.plot(al,lik-min(lik),'k-',label='BAO template')
plt.plot(al,liksm-min(lik),'k:',label='no BAO')
plt.xlabel(r'$\alpha$ (relative isotropic BAO scale)')
plt.ylabel(r'$\Delta\chi^{2}$')
plt.legend()
plt.show()

plt.errorbar(rl,rl**2.*xid,rl**2*np.array(xistd),fmt='ro')
fmod = outdir+'ximodLRG'+wo+str(zmin)+str(zmax)+sbaotemp+str(bs)+'.dat'
fitmod = np.loadtxt(fmod).transpose()
plt.plot(fitmod[0],fitmod[0]**2.*fitmod[1],'k-')
plt.xlim(20,rmax+10)
plt.ylim(-50,100)
plt.show()


#get whatever xi you actually want to test here and replace xiave
lik = bf.doxi_isolike(xiave,cov,mod,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRGEZxiave'+str(zmin)+str(zmax)+sbaotemp+str(bs),diro=outdir)
print('minimum chi2 is '+str(min(lik))+' for '+str(nbin-5)+' dof')
liksm = bf.doxi_isolike(xiave,cov,modsm,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRGEZxiave_smooth_'+str(zmin)+str(zmax)+sbaotemp+str(bs),diro=outdir)
#print(lik)
#print(liksm)
al = [] #list to be filled with alpha values
for i in range(0,len(lik)):
    a = .8+spa/2.+spa*i
    al.append(a)
#below assumes you have matplotlib to plot things, if not, save the above info to a file or something

sigs = sigreg_c12(al,lik)
print('result is alpha = '+str((sigs[2]+sigs[1])/2.)+'+/-'+str((sigs[2]-sigs[1])/2.))


plt.plot(al,lik-min(lik),'k-',label='BAO template')
plt.plot(al,liksm-min(lik),'k:',label='no BAO')
plt.xlabel(r'$\alpha$ (relative isotropic BAO scale)')
plt.ylabel(r'$\Delta\chi^{2}$')
plt.legend()
plt.show()

plt.errorbar(rl,rl**2.*xiave,rl**2*np.array(xistd),fmt='ro')
fmod = outdir+'ximodLRGEZxiave'+str(zmin)+str(zmax)+sbaotemp+str(bs)+'.dat'
fitmod = np.loadtxt(fmod).transpose()
plt.plot(fitmod[0],fitmod[0]**2.*fitmod[1],'k-')
plt.xlim(20,rmax+10)
plt.ylim(-50,100)
plt.show()
                
                