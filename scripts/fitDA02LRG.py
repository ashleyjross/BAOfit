import BAOfit as bf
import numpy as np
import os
from matplotlib import pyplot as plt

rmin = 50
rmax = 150
maxb = 80.
bs = 5.
binc = 0

zmin = 0.8
zmax = 1.1
bs = 4

Nmock = 500

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


def get_xi0cov():
    
    dirm = '/global/project/projectdirs/desi/users/dvalcin/Mocks/'
    fnm = 'xi_lognormal_lrg_sub_'
    xin0 = np.loadtxt(dirm+fnm+'1.txt')
    nbin = len(xin0)
    print(nbin)
    xiave = np.zeros((nbin))
    cov = np.zeros((nbin,nbin))

    Ntot = 0
    fac = 1.
    for i in range(1,Nmock):
        nr = str(i)
        xii = np.loadtxt(dirm+fnm+nr+'.txt').transpose()
        xic = xii[1]
        xiave += xic
        Ntot += 1.
    print( Ntot)        
    xiave = xiave/float(Ntot)
    for i in range(1,Nmock):
        nr = str(i)
        xii = np.loadtxt(dirm+fnm+nr+'.txt').transpose()
        xic = xii[1]
        for j in range(0,nbin):
            xij = xic[j]#-angfac*xiit[j]
            for k in range(0,nbin):
                xik = xic[k]#-angfac*xiit[k]
                cov[j][k] += (xij-xiave[j])*(xik-xiave[k])

    cov = cov/float(Ntot)                   
        
    return cov



datadir =  '/global/cfs/cdirs/desi/survey/catalogs/DA02/LSS/guadalupe/LSScats/2/xi/'

data = datadir+'xi024LRGDA02_'+str(zmin)+str(zmax)+'2_default_FKPlin'+str(bs)+'.dat'
d = np.loadtxt(data).transpose()
xid = d[1]
rl = []
nbin = 0
for i in range(0,len(d[0])):
    r = i*bs+bs/2.+binc
    rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
    rl.append(rbc) 
    if rbc > rmin and rbc < rmax:
        nbin += 1
rl = np.array(rl)
print(rl)
print(xid)
covm = get_xi0cov() #will become covariance matrix to be used with data vector
cfac = 5/4
covm *= cfac**2.
diag = []
for i in range(0,len(covm)):
    diag.append(np.sqrt(covm[i][i]))
diag = np.array(diag)
plt.plot(rl,rl*diag,label='lognormal mocks')
plt.plot(rl,rl*d[4],label='jack-knife')
plt.xlabel('s (Mpc/h)')
plt.ylabel(r's$\sigma$')
plt.legend()
plt.show()

mod = np.loadtxt('BAOtemplates/xi0Challenge_matterpower0.404.08.015.00.dat').transpose()[1]
modsm = np.loadtxt('BAOtemplates/xi0smChallenge_matterpower0.404.08.015.00.dat').transpose()[1]
spa=.001
outdir = os.environ['HOME']+'/DA02baofits/'
lik = bf.doxi_isolike(xid,covm,mod,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRG'+str(zmin)+str(zmax)+'bosspktemp'+str(bs),diro=outdir)
print('minimum chi2 is '+str(min(lik))+' for '+str(nbin-5)+' dof')
liksm = bf.doxi_isolike(xid,covm,modsm,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRG'+str(zmin)+str(zmax)+'bosspktempsm'+str(bs),diro=outdir)
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

plt.errorbar(rl,rl**2.*xid,rl**2*diag,fmt='ro')
fmod = outdir+'ximodLRG'+str(zmin)+str(zmax)+'bosspktemp'+str(bs)+'.dat'
mod = np.loadtxt(fmod).transpose()
plt.plot(mod[0],mod[0]**2.*mod[1],'k-')
plt.xlim(20,rmax+10)
plt.ylim(-50,100)
plt.show()
                