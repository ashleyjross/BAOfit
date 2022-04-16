import BAOfit as bf
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import numpy.linalg as linalg

import pycorr

rmin = 50
rmax = 150
rmaxb = 80.
binc = 0

zmin = 0.8
zmax = 1.1
bs = 4

sfog = 3 #fog velocity term, 3 is kind of cannonical
dperp = 4 # about right for z ~0.8
drad = 8 # about right for z ~0.8

Nmock = 1000



#make BAO template given parameters above, using DESI fiducial cosmology and cosmoprimo P(k) tools
#mun is 0 for pre rec
#sigs is only relevant if mun != 0 and should then be the smoothing scale for reconstructions
#beta is b/f, so should be changed depending on tracer
#sp is the spacing in Mpc/h of the templates that get written out, most of the rest of the code assumes 1
#BAO and nowiggle templates get written out for xi0,xi2,xi4 (2D code reconstructions xi(s,mu) from xi0,xi2,xi4)

gentemp = False
if gentemp:
    bf.mkxifile_3dewig(sp=1.,v='y',pkfile='DESI',mun=0,beta=0.4,sfog=sfog,sigt=dperp,sigr=drad,sigs=15.)

#sys.exit()

#make covariance matrix from EZ mocks
#def get_xi0cov():
znm = str(10*zmin)[:1]+str(10*zmax)[:1]
dirm = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/csaulder/EZmocks/'
fnm = 'EZmock_results_'+znm
result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+'_1.npy')
rebinned = result[:(result.shape[0]//bs)*bs:bs]
ells = (0, 2)
s, xiell = rebinned(ells=ells, return_sep=True)
indmin = 0
indmax = len(s)
indmaxb = len(s)
sm = 0
sx = 0
sxb = 0
for i in range(0,len(s)):
    if s[i] > rmin and sm == 0:
        indmin = i
        sm = 1
    if s[i] > rmax and sx == 0:
        indmax = i
        sx = 1
    if s[i] > rmaxb and sxb == 0:
        indmaxb = i
        sxb = 1
print(indmin,indmax)        
nbin = 2*(indmax-indmin)
print(nbin)
xiave = np.zeros((nbin))
cov = np.zeros((nbin,nbin))
nbinb = 2*(indmaxb-indmin)
print(nbinb)
xiaveb = np.zeros((nbinb))
covb = np.zeros((nbinb,nbinb))

Ntot = 0
fac = 1.
for i in range(1,Nmock+1):
	nr = '_'+str(i)
	result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+nr+'.npy')
	rebinned = result[:(result.shape[0]//bs)*bs:bs]
	xic0 = rebinned(ells=ells)[0][indmin:indmax]
	xic2 = rebinned(ells=ells)[1][indmin:indmax]
	xic = np.concatenate((xic0,xic2))
	xiave += xic
	xic0b = rebinned(ells=ells)[0][indmin:indmaxb]
	xic2b = rebinned(ells=ells)[1][indmin:indmaxb]
	xicb = np.concatenate((xic0b,xic2b))
	xiaveb += xicb
	Ntot += 1.
print( Ntot)        
xiave = xiave/float(Ntot)
xiaveb = xiaveb/float(Ntot)
for i in range(1,Nmock+1):
	nr = '_'+str(i)
	result = pycorr.TwoPointCorrelationFunction.load(dirm+fnm+nr+'.npy')
	rebinned = result[:(result.shape[0]//bs)*bs:bs]
	xic0 = rebinned(ells=ells)[0][indmin:indmax]
	xic2 = rebinned(ells=ells)[1][indmin:indmax]
	xic = np.concatenate((xic0,xic2))
	xic0b = rebinned(ells=ells)[0][indmin:indmaxb]
	xic2b = rebinned(ells=ells)[1][indmin:indmaxb]
	xicb = np.concatenate((xic0b,xic2b))
	for j in range(0,nbin):
		xij = xic[j]
		for k in range(0,nbin):
			xik = xic[k]
			cov[j][k] += (xij-xiave[j])*(xik-xiave[k])
	for j in range(0,nbinb):
		xij = xicb[j]
		for k in range(0,nbinb):
			xik = xicb[k]
			covb[j][k] += (xij-xiaveb[j])*(xik-xiaveb[k])

cov = cov/float(Ntot)             
covb = covb/float(Ntot)      
sc = np.concatenate((s[indmin:indmax],s[indmin:indmax]))
scb = np.concatenate((s[indmin:indmaxb],s[indmin:indmaxb]))
#return cov

#cov = get_xi0cov()
xistd = []
covn = np.zeros((len(xiave),len(xiave)))
for i in range(0,len(xiave)):
     xistd.append(np.sqrt(cov[i][i]))
     for j in range(0,len(xiave)):
         covn[i][j] = cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])
# plt.errorbar(sc,sc**2.*xiave,sc**2.*np.array(xistd))
# plt.show()
# invcov = linalg.inv(cov)
# #plt.imshow(invcov)
# plt.imshow(covn)
# plt.show()
# sys.exit()

invc = np.linalg.pinv(cov) #the inverse covariance matrix to pass to the code
invcb = np.linalg.pinv(covb) #the inverse covariance matrix to get the bias values to pass to the code
#mod = 'Challenge_matterpower0.5933.058.515.00.dat' #BAO template used		
#fout = 'desi_challeng1_ajr_prerec_0.5933.058.515.00'
wm = str(sfog)+str(dperp)+str(drad)
mod = 'DESI0.4'+wm+'15.00.dat'


#bias priors, log around best fit up to rmaxb
Bp = 100#0.4
Bt = 100#0.4

spa = .001
mina = .8
maxa = 1.2
outdir = os.environ['HOME']+'/DESImockbaofits/'

rl = []
#nbin = 0
for i in range(0,len(sc)):
    r = sc[i]
    #correct for pairs should have slightly larger average pair distance than the bin center
    #this assumes mid point of bin is being used and pairs come from full 3D volume
    rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) 
    rl.append(rbc) 

rlb = []
#nbin = 0
for i in range(0,len(scb)):
    r = scb[i]
    #correct for pairs should have slightly larger average pair distance than the bin center
    #this assumes mid point of bin is being used and pairs come from full 3D volume
    rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) 
    rlb.append(rbc) 


abdir = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Xi/jmena/'
xid0 = np.loadtxt(abdir+'Xi_0_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[0][indmin:indmax] #just look at first mock
xid2 = np.loadtxt(abdir+'Xi_2_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[0][indmin:indmax] #just look at first mock
xid = np.concatenate((xid0,xid2))

xid0b = np.loadtxt(abdir+'Xi_0_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[0][indmin:indmaxb] #just look at first mock
xid2b = np.loadtxt(abdir+'Xi_2_zmin'+str(zmin)+'_zmax'+str(zmax)+'.txt').transpose()[0][indmin:indmaxb] #just look at first mock
xidb = np.concatenate((xid0b,xid2b))
fout = 'LRGabcutsky0'+str(zmin)+str(zmax)+wm+str(bs)
bf.Xism_arat_1C_an(xid,invc,rl,mod,xidb,invcb,rlb,verbose=True,Bp=Bp,Bt=Bt,fout=fout,dirout=outdir)
bf.plot_2dlik(os.environ['HOME']+'/DESImockbaofits/2Dbaofits/arat'+fout+'1covchi.dat')

sys.exit()

fout = 'LRGEZxiave'+str(zmin)+str(zmax)+wm+str(bs)
bf.Xism_arat_1C_an(xiave,invc,rl,mod,xiaveb,invcb,rlb,verbose=True,Bp=Bp,Bt=Bt,fout=fout,dirout=outdir)


rl = np.array(rl)
sbaotemp = str(sfog)+str(dperp)+str(drad)
mod = np.loadtxt('BAOtemplates/xi0DESI0.4'+sbaotemp+'15.00.dat').transpose()[1]
modsm = np.loadtxt('BAOtemplates/xi0smDESI0.4'+sbaotemp+'15.00.dat').transpose()[1]

spa=.001


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
                
                