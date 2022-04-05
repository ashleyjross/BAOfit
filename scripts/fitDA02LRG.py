import BAOfit as bf
import numpy as np
import os

rmin = 50
rmax = 150
maxb = 80.
bs = 5.
binc = 0

zmin = 0.8
zmax = 1.1
bs = 4

Nmock = 500

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
    cov = cov/float(Ntot)                   
        
    return cov



datadir =  '/global/cfs/cdirs/desi/survey/catalogs/DA02/LSS/guadalupe/LSScats/2/xi/'

data = datadir+'xi024LRGDA02_'+str(zmin)+str(zmax)+'2_default_FKPlin'+str(bs)+'.dat'
d = np.loadtxt(data).transpose()
xid = d[1]
rl = []
for i in range(0,len(d[0])):
    r = i*bs+bs/2.+binc
    rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
    rl.append(rbc) 

covm = get_xi0cov() #will become covariance matrix to be used with data vector

mod = np.loadtxt('BAOtemplates/xi0Challenge_matterpower0.404.08.015.00.dat').transpose()[1]
modsm = np.loadtxt('BAOtemplates/xi0smChallenge_matterpower0.404.08.015.00.dat').transpose()[1]

lik = bf.doxi_isolike(xid,covm,mod,modsm,rl,bs=bs,rmin=50,rmax=150,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=80.,spa=.001,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRG'+str(zmin)+str(zmax)+'bosspktemp'+str(bs),diro=os.environ['HOME']+'/DA02baofits/')
liksm = bf.doxi_isolike(xid,covm,modsm,modsm,rl,bs=bs,rmin=50,rmax=150,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=80.,spa=.001,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo='LRG'+str(zmin)+str(zmax)+'bosspktemp'+str(bs),diro=os.environ['HOME']+'/DA02baofits/')

al = [] #list to be filled with alpha values
for i in range(0,len(lik)):
    a = .8+spa/2.+spa*i
    al.append(a)
#below assumes you have matplotlib to plot things, if not, save the above info to a file or something
from matplotlib import pyplot as plt
plt.plot(al,lik-min(cl),'k-')
plt.plot(al,liksm-min(cl),'k:')
plt.show()
                