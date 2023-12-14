'''
one code for all BAO fitting, copying from LSSanalysis and improving documentation
'''

from numpy import loadtxt
from math import *
from EH import simulate
from scipy.special import jn
import numpy as np
from fftlog_JB import *
import numpy.linalg as linalg
from numpy import zeros,dot
from numpy.linalg import pinv



'''
first, functions to make BAO templates
linear P(k) should be in powerspectra directory
output goes to BAOtemplates directory
'''

def P2(mu):
    return .5*(3.*mu**2.-1.)
    
def P4(mu):
    return .125*(35.*mu**4.-30.*mu**2.+3.)

def findPolya(H,ci,d):
    #analytically solves for best-fit polynomial nuisance terms, given BAO and bias parameters
    #http://pdg.lbl.gov/2016/reviews/rpp2016-rev-statistics.pdf eq. 39.22 and surrounding
    ht = H.transpose()
    onei = linalg.pinv(np.dot(np.dot(H,ci),ht))
    comb = np.dot(np.dot(onei,H),ci)
    return np.dot(comb,d)


def mkxifile_3dewig(sp=1.,v='y',pkfile='Challenge_matterpower',mun=0,beta=0.4,sfog=0,sigz=0,sigt=6.,sigr=10.,sigs=15.):
    '''
    create xi0,xi2,xi4 BAO and no BAO templates for use in BAO fitting code
    the templates are linear+BAO damping with bias = 1 at z = 0
    just multiply by (bD[z])^2 and amplitudes should be consistent with chosen linear bias at redshift z
    variables:
    sp is spacing, default is 1 mpc/h
    v is printing things, right now only r
    pkfile is input linear P(k)
    mun = 0 for pre rec and 1 for post-rec with RSD removal (controls effect of smoothing scaling on RSD)
    beta is fiducial f/b
    sfog: the streaming velocity parameter, often denoted Sigma_s
    sigz: for redshift uncertainties, ignore for most cases
    sigt: transverse BAO damping, Sigma_\perp
    sigr: radial BAO damping, Sigma_||
    sigs: smoothing scale used in reconstruction (irrelevant if mun = 0)
    '''
    dirout = 'BAOtemplates/'
    wsigz = ''
    if sigz != 0:
        wsigz += 'sigz'+str(sigz)
    #generate power spectra 
    k,pl0,pl2,pl4,psm0,psm2,psm4 = pk3elldfile_dewig(file=pkfile,beta=beta,sfog=sfog,sigz=sigz,sigt=sigt,sigr=sigr,mun=mun,sigs=sigs,pw='y')    
    #open files for writing
    fout = pkfile+str(beta)+str(sfog)+str(sigt)+str(sigr)+str(sigs)+wsigz+str(mun)+'.dat'
    f0 = open(dirout+'xi0'+fout,'w')
    f2 = open(dirout+'xi2'+fout,'w')
    f4 = open(dirout+'xi4'+fout,'w')
    f0mc = open(dirout+'xi0sm'+fout,'w')
    f2mc = open(dirout+'xi2sm'+fout,'w')
    f4mc = open(dirout+'xi4sm'+fout,'w')
    #make r array
    rl = []
    r = 10.
    while r < 300:
        rl.append(r)
        r += sp
        if v == 'y':
            print(r)
    rl = np.array(rl)
    #transform to get xi 
    rout,xiout0 = HankelTransform(k,pl0,q=1.5,mu=0.5,output_r=rl,output_r_power=-3,r0=10.)
    rout,xiout2 = HankelTransform(k,pl2,q=1.5,mu=2.5,output_r=rl,output_r_power=-3,r0=10.)
    rout,xiout4 = HankelTransform(k,pl4,q=1.5,mu=4.5,output_r=rl,output_r_power=-3,r0=10.)
    #sm are no BAO templates
    rout,xiout0sm = HankelTransform(k,psm0,q=1.5,mu=0.5,output_r=rl,output_r_power=-3,r0=10.)
    rout,xiout2sm = HankelTransform(k,psm2,q=1.5,mu=2.5,output_r=rl,output_r_power=-3,r0=10.)
    rout,xiout4sm = HankelTransform(k,psm4,q=1.5,mu=4.5,output_r=rl,output_r_power=-3,r0=10.)
    

    #write them out
    for i in range(0,len(rout)):
        f0.write(str(rout[i])+' '+str(xiout0[i]/(5.*pi))+'\n')
        f2.write(str(rout[i])+' '+str(xiout2[i]/(-5.*pi))+'\n')
        f4.write(str(rout[i])+' '+str(xiout4[i]/(5.*pi))+'\n')
        f0mc.write(str(rout[i])+' '+str(xiout0sm[i]/(5.*pi))+'\n')
        f2mc.write(str(rout[i])+' '+str(xiout2sm[i]/(-5.*pi))+'\n')
        f4mc.write(str(rout[i])+' '+str(xiout4sm[i]/(5.*pi))+'\n')

        
    f0.close()
    f2.close()
    f4.close()
    f0mc.close()
    f2mc.close()
    f4mc.close()

    return True

def pk3elldfile_dewig(file='Challenge_matterpower',beta=0.4,sigt=3.0,sigr=3.0,sfog=3.5,mun=1.,sigs=15.,ns=.963,sigz=0,pw='y'):
    '''
    returns arrays for k, p0,2,4 multipoles for BAO and no BAO
    file is input linear P(k)
    
    mun = 0 for pre rec and 1 for post-rec with RSD removal (controls effect of smoothing scaling on RSD)
    beta is fiducial f/b
    sfog: the streaming velocity parameter, often denoted Sigma_s
    sigz: for redshift uncertainties, ignore for most cases
    sigt: transverse BAO damping, Sigma_\perp
    sigr: radial BAO damping, Sigma_||
    sigs: smoothing scale used in reconstruction (irrelevant if mun = 0)
    ns: varying ns used in no BAO P(k); best results often come from using one that is slightly different than linear P(k) one (probably could be fixed with EH.py)
    
    '''
    from scipy.integrate import quad
    from Cosmo import distance
    mult = 1.
    dir = 'powerspectra/'
    if file=='Challenge_matterpower' or file == 'TSPT_out':
        om = 0.31
        #lam = 1
        h = .676
        nindex = ns
        ombhh = .022
    if file == 'MICE_matterpower':
        om = 0.25
        #lam = .75
        h = .7
        ombhh = .044*0.7*.7 
        nindex = .949
    if file == 'DESI':
        from cosmoprimo.fiducial import DESI
        from cosmoprimo import PowerSpectrumBAOFilter
        cosmo = DESI()
        pkz = cosmo.get_fourier().pk_interpolator()
        pk = pkz.to_1d(z=0)
        kl = np.loadtxt(dir+'Challenge_matterpower.dat').transpose()[0] #this k spacing is known to work well
        pkv = pk(kl)
        pknow = PowerSpectrumBAOFilter(pk, engine='wallish2018').smooth_pk_interpolator()
        pksmv = pknow(kl)
    if file == 'BOSS':
        from cosmoprimo.fiducial import BOSS
        from cosmoprimo import PowerSpectrumBAOFilter
        cosmo = BOSS()
        pkz = cosmo.get_fourier().pk_interpolator()
        pk = pkz.to_1d(z=0)
        kl = np.loadtxt(dir+'Challenge_matterpower.dat').transpose()[0] #this k spacing is known to work well
        pkv = pk(kl)
        pknow = PowerSpectrumBAOFilter(pk, engine='wallish2018').smooth_pk_interpolator()
        pksmv = pknow(kl)
    
    if file != 'DESI' and file != 'BOSS':
        f = np.loadtxt(dir+file+'.dat').transpose()
    if pw == 'y':
        fo = open('P02'+file+'beta'+str(beta)+'sigs'+str(sfog)+'sigxy'+str(sigt)+'sigz'+str(sigr)+'Sk'+str(sigs)+'.dat','w')
        fo.write('# k P0 P2 P4 Psmooth0 Psmooth2 Psmooth4 Plin Plinsmooth\n')
    if file != 'Pk_MICEcosmology_z0_Plin_Pnowig' and file != 'DESI':
        s = simulate(omega=om,lamda=1-om,h=h,nindex=nindex,ombhh=ombhh)
    elif file == 'Pk_MICEcosmology_z0_Plin_Pnowig':
        mult = 8.*pi**3.    
    pl2 = []
    pl4 = []
    beta0 = 0.4
    for i in range(0,100):
        pl2.append(P2(i/100.+.005))
    for i in range(0,100):
        pl4.append(P4(i/100.+.005))
    mul = []
    anipolyl = []
    for i in range(0,100):
        mu = i/100.+.005
        mul.append(mu)      
    b = 2.
    ff =beta*b
    if sigz != 0:
        z = .8
        d = distance(.25,.75)
        sigzc = d.cHz(z)*sigz
    
    norm = 1
    if file != 'DESI':
        kl = f[0]
        pml = f[1]
        norm = pml[0]/s.Psmooth(kl[0],0)
    p0l = []
    p2l = []
    p4l = []
    psm0l = []
    psm2l = []
    psm4l = []

    for i in range(0,len(kl)):
        k = kl[i]
        if file !='DESI':
            pk = pml[i]*mult
        else:
            pk = pkv[i]
        pk0 = 0
        pk2 = 0
        pk4 = 0
        pksm0 = 0
        pksm2 = 0
        pksm4 = 0
        if file == 'Pk_MICEcosmology_z0_Plin_Pnowig':
            pksm = float(f[i].split()[2])*mult
        elif file == 'DESI':
            pksm = pksmv[i]
        else:   
            pksm = s.Psmooth(k,0)*norm
        dpk = pk-pksm
        for m in range(0,100):
            #mu = (1.-mul[m])           
            mu = mul[m]

            if mun == 'n':
                mu = (1.-mul[m])
            if sfog > 0:
                F = 1./(1.+k**2.*mu**2.*sfog**2./2.)**2.
            else:
                F = (1.+k**2.*sfog**2./2.)**2./(1.+k**2.*(1.-mu)**2.*sfog**2./2.)**2.
            if mun == 'b':
                mus2 = mu**2.
                F = (1.+beta*mus2*(1.-exp(-0.5*(k*sigs)**2.)))**2.*1./(1.+k**2.*mu**2.*(sfog)**2./2.)**2.
            #C *doesn't include damping*
            S = mun*exp(-0.5*(k*sigs)**2.)
            C = (1.+beta*mu*mu*(1.-S))*1./(1.+k**2.*mu**2.*(sfog)**2./2.)
            sigv2 = (1-mu**2.)*sigt**2./4.+mu**2.*sigr**2./4.
            damp = exp(-1.*k*k*sigv2)
            if sigz != 0:
                C = C*exp(-0.5*k*k*mu*mu*sigzc*sigzc)   
            pkmu = C**2.*(dpk*damp**2.+pksm)
            pkmusm = C**2.*pksm
            pk0 += pkmu
            pk2 += pkmu*pl2[m]
            pk4 += pkmu*pl4[m]
            pksm0 += pkmusm
            pksm2 += pkmusm*pl2[m]
            pksm4 += pkmusm*pl4[m]
        pk0 = pk0/100.
        pk2 = 5.*pk2/100.
        pk4 = 9.*pk4/100.
        pksm0 = pksm0/100.
        pksm2 = 5.*pksm2/100.
        pksm4 = 9.*pksm4/100.
        p0l.append(pk0)
        p2l.append(pk2)
        p4l.append(pk4)
        psm0l.append(pksm0)
        psm2l.append(pksm2)
        psm4l.append(pksm4)
        print(pk0,pksm0,dpk)

        if pw == 'y':
            fo.write(str(k)+' '+str(pk0)+' '+str(pk2)+' '+str(pk4)+' '+str(pksm0)+' '+str(pksm2)+' '+str(pksm4)+' '+str(pk)+' '+str(pksm)+'\n')
        
    if pw == 'y':
        fo.close()
        from matplotlib import pyplot as plt
        from numpy import loadtxt as load
        d = load('P02'+file+'beta'+str(beta)+'sigs'+str(sfog)+'sigxy'+str(sigt)+'sigz'+str(sigr)+'Sk'+str(sigs)+'.dat').transpose()
        plt.xlim(0,.5)
        plt.plot(d[0],d[-2]/d[-1])
        plt.plot(d[0],np.ones(len(d[0])),'--')
        plt.show()
    p0l = np.array(p0l)
    p2l = np.array(p2l)
    p4l = np.array(p4l)
    psm0l = np.array(psm0l)
    psm2l = np.array(psm2l)
    psm4l = np.array(psm4l)
    return kl,p0l,p2l,p4l,psm0l,psm2l,psm4l


class baofit_iso:
    def __init__(self,xid,covd,modl,modsmoothl,rl,bs=8,rmin=50,rmax=150,sp=1.,facc=1.):
        #xid and covd are the data vector and the covariance matrix and should be matched in length
        #modl is the BAO template, assumed to have spacing sp between 10 and 300 mpc/h
        #rl is the list of r values matching xid and covd
        self.xim = [] #these lists will be filled based on the r limits
        self.rl = []
        self.sp = sp
        bs = float(bs)#rl[1]-rl[0] #assumes linear bins
        s = 0
        nxib = len(xid)
        rsw = 0
        Bnode = 50.
        print( 'total available xi(s) bins '+str(nxib))
        for i in range(0,nxib):
            r = rl[i]
            if r > rmin and r < rmax:               
                if s == 0:
                    mini = i
                    s = 1
                #rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.)
                rbc = r
                #rbc = i*bs+bs/2.
                #print(rbc,r,bs)
                #this is the properly weighted bin center assume spherical symmetry
                self.rl.append(rbc)
                self.xim.append(xid[i])
        self.nbin = len(self.rl)
        print ('using '+ str(self.nbin)+' xi(s) bins')
        mt = covd[mini:mini+self.nbin,mini:mini+self.nbin]
        self.invt = linalg.pinv(mt)
        self.ximodmin = 10. #minimum of template
        self.modl = modl
        #print(self.invt)
        self.modsmoothl = modsmoothl
                        
    def wmod(self,r):
        sp = self.sp
        indd = int((r-self.ximodmin)/sp)
        indu = indd + 1
        fac = (r-self.ximodmin)/sp-indd
        if fac > 1.:
            print ('BAD FAC in wmod')
            return 'ERROR, BAD FAC in wmod'
        if indu >= len(self.modl)-1:
            #return -5.47608128044e-05,5.7422824622e-06
            return self.modl[-1]
        #print(self.modl[indu],fac,self.modl[indd],indu,indd)
        a = self.modl[indu]*fac+(1.-fac)*self.modl[indd]
        return a

    def wmodsmooth(self,r):
        sp = self.sp
        indd = int((r-self.ximodmin)/sp)
        indu = indd + 1
        fac = (r-self.ximodmin)/sp-indd
        if fac > 1.:
            print ('BAD FAC in wmod')
            return 'ERROR, BAD FAC in wmod'
        if indu >= len(self.modsmoothl)-1:
            #return -5.47608128044e-05,5.7422824622e-06
            return self.modsmoothl[-1]
        a = self.modsmoothl[indu]*fac+(1.-fac)*self.modsmoothl[indd]
        return a
        
        

    def chi_templ_alphfXX(self,list,Bnode=50.,wo='n',fw='',diro=''):
        from time import time
        t = time()
        B = list[0]
        if B < 0:
            return 1000
        alph = self.alph
        #A0 = list[1]
        #A1 = list[2]
        #A2 = list[3]
        pv = []
        for i in range(0,self.nbin):
            r = self.rl[i]*alph
            wm = self.wmod(r)
            #if len(self.xim[i]-B*wm) > 1:
            #print self.xim[i]-B*wm,self.xim[i],B,wm
            pv.append(self.xim[i]-B*wm)
         
        Al = findPolya(self.H,self.invt,pv)
        #A0,A1,A2 = Al[0],Al[1],Al[2]
        #self.A0 = A0
        #self.A1 = A1
        #self.A2 = A2
        modl = np.zeros((self.nbin))
        if wo == 'y':
            fo = open(diro+'ximod'+fw+'.dat','w')
        for i in range(0,self.nbin):
            r = self.rl[i]
            #ply = A0+A1/r+A2/r**2.
            ply = 0
            for j in range(0,self.np):
                ply += Al[j]/r**j
            r = self.rl[i]*alph
            wm = self.wmod(r)
            wsm = self.wmodsmooth(r)
            mod = B*wm+ply
            modsm = B*wsm+ply
            if wo == 'y':
                fo.write(str(self.rl[i])+' '+str(mod)+' '+str(modsm)+' '+str(self.xim[i])+'\n')
                #print mod,self.xim[i]
            modl[i] = mod
        if wo == 'y':
            fo.close()      
            
        ChiSq = 0
        chid = 0
        Chioff = 0
        dl = np.zeros((self.nbin))
        
        for i in range(0,self.nbin):
            dl[i] = self.xim[i]-modl[i]
        chit = np.dot(np.dot(dl,self.invt),dl)
        BBfac = (log(B/self.BB)/self.Bp)**2. #bias prior
        return chit+BBfac

    def chi_templ_alphfXXn(self,list,wo='n',fw=''):
        #chi2 with no polynomial
        from time import time
        t = time()
        B = list[0]
        if B < 0:
            return 1000
        alph = self.alph
        modl = np.zeros((self.nbin))
        if wo == 'y':
            fo = open('ximod'+fw+'.dat','w')
        for i in range(0,self.nbin):
            r = self.rl[i]*alph
            wm = self.wmod(r)
            mod = B*wm
            modl[i] = mod
        if wo == 'y':
            fo.close()      
        
        
        dl = np.zeros((self.nbin))  
        for i in range(0,self.nbin):
            dl[i] = self.xim[i]-modl[i]
        #print(self.xim,modl,dl)
        chit = np.dot(np.dot(dl,self.invt),dl)  
        return chit


def doxi_isolike(xid,covd,modl,modsmoothl,rl,bs=8,rmin=50,rmax=150,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=50.,spa=.001,mina=.8,maxa=1.2,Nmock=1000,v='',wo='',diro=''):
    '''
    1D fit to monopole for alpha_iso
    returns list of chi2(alpha)
    xid is array for data
    covd is the cov matrix
    modl is the BAO template
    modsmoothl is the no BAO template
    rl is the array of r bin centers
    bs is the bin size for xi
    rmin is the minimum for the bin center
    rmax is maximum of the bin center
    npar is the number of polynomial terms (might only work for 3 or 0)
    sp is spacing of the BAO template
    Bp is size of the log(B/Bbest) prior
    rminb is the minimum bin center for finding Bbest
    rmaxb is the maximum bin center for finding Bbest
    spa is the spacing for the alpha grid
    mina is the minimum alpha
    maxa is the maximum alpha
    Nmock is the number of mocks used to create the cov matrix
    wo is string for writing output files
    v = 'y' will print info
    diro is directory for output
    '''
    #
    from time import time
    from optimize import fmin
    print (np)
    b = baofit_iso(xid,covd,modl,modsmoothl,rl,rmin=rmin,rmax=rmax,sp=sp,bs=bs)
    b.Bp = Bp
    b.np = npar
    b.H = np.zeros((npar,b.nbin))
    chi2fac = (Nmock-b.nbin-2.)/(Nmock-1.)
    print (b.nbin,chi2fac)
    for i in range(0,b.nbin):
        for j in range(0,npar):
            b.H[j][i] = 1./b.rl[i]**j
    if rmin == rmaxb:
        rmaxb += (b.rl[1]-b.rl[0])*1.1 #increase rmaxb by one bin size if set poorly
    bb = baofit_iso(xid,covd,modl,modsmoothl,rl,rmin=rmin,rmax=rmaxb,sp=sp,bs=bs)
    #bb is to set bias prior
    bb.np = npar
    bb.H = np.zeros((npar,bb.nbin))
    for i in range(0,bb.nbin):
        for j in range(0,npar):
            bb.H[0][i] = 1./bb.rl[i]**j

    alphl= []
    chil = []
    likl = []
    chim = 1000
    na = int((maxa-mina)/spa)
    likm = -1
    pt = 0
    A0 = 0
    A1 = 1.
    A2 = 0
    b.alph = 1.
    bb.alph = b.alph
    #b.alph = 0
    B = .1
    chiBmin = 1000
    Bmax = 10.
    #simple search for best-fit B
    while B < Bmax:
        bl = [B]
        chiB = bb.chi_templ_alphfXXn(bl)*chi2fac
            
        if chiB < chiBmin:
            chiBmin = chiB
            BB = B
        B += .01    
    print ('best-fit bias factor is '+str(BB)+' '+str(chiBmin))
    if BB >= Bmax-.011:
        print( 'WARNING, best-fit bias is at max tested value')
    b.BB = BB       
    #b.BB = 1. #switch to this to make bias prior centered on input rather than fit value
    B = BB
    for i in range(0,na):       
        b.alph = mina+spa*i+spa/2.
        #inl = np.array([B,A0,A1,A2]) #from older version without solving for nuisance terms analytically
        inl = B
        #(B,A0,A1,A2) = fmin(b.chi_templ_alphfXX,inl,disp=False) #from older version without solving for nuisance terms analytically
        if npar > 0:
            B = fmin(b.chi_templ_alphfXX,inl,disp=False)
            chi = b.chi_templ_alphfXX((B))*chi2fac
        else:
            B = fmin(b.chi_templ_alphfXXn,inl,disp=False)
            chi = b.chi_templ_alphfXXn((B))*chi2fac
        #chi = b.chi_templ_alphfXX((B,A0,A1,A2))*chi2fac #from older version without solving for nuisance terms analytically
        
        if v == 'y':
            print (b.alph,chi,B[0],b.A0[0],b.A1[0],b.A2[0]) #single values getting output as arrays, silly, but works so not worrying about it
        alphl.append(b.alph)
        chil.append(chi)
        if chi < chim:
            chim = chi
            alphm = b.alph
            Bm = B
            #A0m = b.A0 #from older version without solving for nuisance terms analytically
            #A1m = b.A1
            #A2m = b.A2
    b.alph = alphm
    if npar > 0:    
        b.chi_templ_alphfXX((Bm),wo='y',fw=wo,diro=diro)
    return chil

'''
2D baofit fits for alpha_||/alpha_perp
'''

class baofit3D_ellFull_1cov:
    def __init__(self,dv,ic,mod,rl,dirout=''):
        self.xim = dv
        self.dirout = dirout
        xinl = []
        xisl = []
        xinli = []
        xisli = []
        self.rl = rl
        s = 0
        m2 = 1.
        #self.B0fac = (1.+2/3.*B0+.2*B0**2.)
        #self.B2fac = (4/3.*B0+4/7.*B0**2.)
        #self.B4fac = 8/35.*B0**2.

        self.nbin = len(self.rl)
        
        #print(self.nbin)
        #print(self.xim)        
        #print(self.rl)
        self.invt = ic
        if self.nbin != len(self.invt):
            print('vector matrix mismatch!')
            return 'vector matrix mismatch!'

        self.ximodmin = 10.

        self.x0 = [] #empty lists to be filled for model templates
        self.x2 = []
        self.x4 = []
        self.x0sm = []
        self.x2sm = []
        self.x4sm = []
        
        mf0 = open('BAOtemplates/xi0'+mod).readlines()
        mf2 = open('BAOtemplates/xi2'+mod).readlines()
        mf4 = open('BAOtemplates/xi4'+mod).readlines()
        mf0sm = open('BAOtemplates/xi0sm'+mod).readlines()
        mf2sm = open('BAOtemplates/xi2sm'+mod).readlines()
        mf4sm = open('BAOtemplates/xi4sm'+mod).readlines()
        for i in range(0,len(mf0)):
            ln0 = mf0[i].split()
            self.x0.append(2.1*(float(ln0[1])))
            self.x0sm.append(2.1*float(mf0sm[i].split()[1]))
            ln2 = mf2[i].split()
            m2 = 2.1*(float(ln2[1]))
            self.x2.append(m2)
            self.x2sm.append(2.1*float(mf2sm[i].split()[1]))
            ln4 = mf4[i].split()
            m4 = 2.1*(float(ln4[1]))
            self.x4.append(m4)
            self.x4sm.append(2.1*float(mf4sm[i].split()[1]))
        self.at = 1.
        self.ar = 1.
        self.b0 = 1.
        self.b2 = 1.
        self.b4 = 1.
        r = 20.
        self.H = zeros((6,self.nbin))
        for i in range(0,self.nbin):
            if i < self.nbin/2:
                self.H[0][i] = 1.
                self.H[1][i] = 1./self.rl[i]
                self.H[2][i] = 1./self.rl[i]**2.
            if i >= self.nbin/2:
                self.H[3][i] = 1.
                self.H[4][i] = 1./self.rl[i]
                self.H[5][i] = 1./self.rl[i]**2.
        
    def wmod(self,r,sp=1.):
        self.sp = sp
        sum = 0
        sum2 = 0
        nmu = 100
        dmu = 1./float(nmu)
        for i in range(0,nmu):
            mu = i*dmu+dmu/2.
            al = sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            #ximu = self.b0*self.lininterp(self.x0,rp)+self.b2*P2(mup)*self.lininterp(self.x2,rp)+self.b4*P4(mup)*self.lininterp(self.x4,rp)
            ximu = self.lininterp(self.x0,rp)+P2(mup)*self.lininterp(self.x2,rp)+P4(mup)*self.lininterp(self.x4,rp)
            sum += ximu
            #sum2 += P2(mu)*ximu
            sum2 += mu**2.*ximu
        return dmu*sum,1.5*dmu*sum2#3.*dmu*sum2

    def wmodsm(self,r,sp=1.):
        self.sp = sp
        sum = 0
        sum2 = 0
        nmu = 100
        dmu = 1./float(nmu)
        for i in range(0,nmu):
            mu = i*dmu+dmu/2.
            al = sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            ximu = self.lininterp(self.x0sm,rp)+P2(mup)*self.lininterp(self.x2sm,rp)+P4(mup)*self.lininterp(self.x4sm,rp)
            sum += ximu
            #sum2 += P2(mu)*ximu
            sum2 += mu**2.*ximu
        return dmu*sum,1.5*dmu*sum2#3.*dmu*sum2

    def wmodW(self,r,sp=1.):
        self.sp = sp
        sum = 0
        sum2 = 0
        nmu = 100
        dmu = 1./float(nmu)
        nspl = int(nmu*self.Wsp)
        for i in range(0,nspl):
            mu = i*dmu+dmu/2.
            al = sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            #ximu = self.b0*self.lininterp(self.x0,rp)+self.b2*P2(mup)*self.lininterp(self.x2,rp)+self.b4*P4(mup)*self.lininterp(self.x4,rp)
            ximu = self.lininterp(self.x0,rp)+P2(mup)*self.lininterp(self.x2,rp)+P4(mup)*self.lininterp(self.x4,rp)
            sum += ximu
        for i in range(nspl,nmu):
            mu = i*dmu+dmu/2.
            al = sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            #ximu = self.b0*self.lininterp(self.x0,rp)+self.b2*P2(mup)*self.lininterp(self.x2,rp)+self.b4*P4(mup)*self.lininterp(self.x4,rp)
            ximu = self.lininterp(self.x0,rp)+P2(mup)*self.lininterp(self.x2,rp)+P4(mup)*self.lininterp(self.x4,rp)
            sum2 += ximu
        n0 = 1./float(self.Wsp)
        n2 = 1./(1.-self.Wsp)
        return dmu*sum*n0,dmu*sum2*n2

            
    def lininterp(self,f,r):
        indd = int((r-self.ximodmin)/self.sp)
        indu = indd + 1
        fac = (r-self.ximodmin)/self.sp-indd
        if fac > 1.:
            print('BAD FAC in wmod')
            return 'ERROR'
        if indu >= len(f)-1:
            return 0
        return f[indu]*fac+(1.-fac)*f[indd] 

    def mkxi(self):
        self.xia = []
        for i in range(0,len(self.rl)//2):
            xi0,xi2 = self.wmod(self.rl[i])
            self.xia.append(xi0)
            #self.xi2a.append(xi2)
        for i in range(len(self.rl)//2,self.nbin):
            xi0,xi2 = self.wmod(self.rl[i])
            self.xia.append(xi2)
            #self.xi2a.append(xi2)
        return True 

    def mkxism(self):
        self.xiasm = []
        for i in range(0,len(self.rl)//2):
            xi0,xi2 = self.wmodsm(self.rl[i])
            self.xiasm.append(xi0)
            #self.xi2a.append(xi2)
        for i in range(len(self.rl)//2,self.nbin):
            xi0,xi2 = self.wmodsm(self.rl[i])
            self.xiasm.append(xi2)
            #self.xi2a.append(xi2)
        return True 

    def mkxiW(self):
        self.xi0a = []
        self.xi2a = []
        for i in range(0,len(self.rl)):
            xi0,xi2 = self.wmodW(self.rl[i])
            self.xi0a.append(xi0)
            self.xi2a.append(xi2)
        return True 
        
    def chi_templ_alphfXX(self,list,wo='n',fw='',v='n'):
        from time import time
        t = time()
        BB = list[0]
        if BB < 0:
            return 1000
        A0 = list[1]
        A1 = list[2]
        A2 = list[3]
        Beta = list[4]
        if Beta < 0:
            return 1000
        A02 = list[5]
        A12 = list[6]
        A22 = list[7]
        #self.b0 = BB*(1.+2/3.*Beta+.2*Beta**2.)/self.B0fac
        #self.b2 = BB*(4/3.*Beta+4/7.*Beta**2.)/self.B2fac
        #self.b4 = 8/35.*BB*Beta**2./self.B4fac
        modl = []
        if wo == 'y':
            fo = open('ximod'+fw+'.dat','w')
        for i in range(0,self.nbin//2):
            r = self.rl[i]
            
            mod0 = BB*self.xia[i]+A0+A1/r+A2/r**2.
            modl.append(mod0)
            if wo == 'y':
                fo.write(str(self.rl[i])+' '+str(mod0)+'\n')
        
        for i in range(self.nbin//2,self.nbin):
            r = self.rl[i]
            mod2 = 5.*(Beta*self.xia[i]-BB*0.5*self.xia[i-self.nbin//2])+A02+A12/r+A22/r**2.
            if wo == 'y':
                fo.write(str(self.rl[i])+' '+str(mod2)+'\n')            
            modl.append(mod2)
        if wo == 'y':
            fo.close()      
            

        ChiSq = 0
        chid = 0
        Chioff = 0
        dl = []
        for i in range(0,self.nbin):
            dl.append(self.xim[i]-modl[i])
        chit = dot(dot(dl,self.invt),dl)
        if v == 'y':
            print(dl,chit)
        BBfac = (log(BB/self.BB)/self.Bp)**2.
        #Btfac = ((Beta-self.B0)/self.Bt)**2.
        Btfac = (log(Beta/self.B0)/self.Bt)**2.
        return chit+BBfac+Btfac

    def chi_templ_alphfXX_an(self,list,wo='n',fw='',v='n'):
        from time import time
        t = time()
        BB = list[0]
        if BB < 0:
            return 1000
        #A0 = list[1]
        #A1 = list[2]
        #A2 = list[3]
        Beta = list[1]
        if Beta < 0:
            return 1000
        #A02 = list[5]
        #A12 = list[6]
        #A22 = list[7]
        #self.b0 = BB*(1.+2/3.*Beta+.2*Beta**2.)/self.B0fac
        #self.b2 = BB*(4/3.*Beta+4/7.*Beta**2.)/self.B2fac
        #self.b4 = 8/35.*BB*Beta**2./self.B4fac
        nrbin = self.nbin//2
        modl = []
        if wo == 'y':
            fo = open(self.dirout+'ximod'+fw+'.dat','w')
            fp = open(self.dirout+'xipar'+fw+'dat','w')
        pv = []
        for i in range(0,self.nbin//2):
            pv.append(self.xim[i]-BB*self.xia[i])
        for i in range(self.nbin//2,self.nbin):
            pv.append(self.xim[i]-(5.*(Beta*self.xia[i]-BB*0.5*self.xia[i-self.nbin//2])))
         
        Al = findPolya(self.H,self.invt,pv)
        A0,A1,A2,A02,A12,A22 = Al[0],Al[1],Al[2],Al[3],Al[4],Al[5]
        if wo == 'y':
            fp.write(str(BB)+' '+str(Beta)+' '+str(A0)+' '+str(A1)+' '+str(A2)+' '+str(A02)+' '+str(A12)+' '+str(A22)+'\n')
        for i in range(0,self.nbin//2):
            r = self.rl[i]
            
            mod0 = BB*self.xia[i]+A0+A1/r+A2/r**2.
            modl.append(mod0)
            if wo == 'y':
                mod0sm = BB*self.xiasm[i]+A0+A1/r+A2/r**2.
                fo.write(str(self.rl[i])+' '+str(mod0)+' '+str(mod0sm)+'\n')
        
        for i in range(self.nbin//2,self.nbin):
            r = self.rl[i]
            mod2 = 5.*(Beta*self.xia[i]-BB*0.5*self.xia[i-nrbin])+A02+A12/r+A22/r**2.
            if wo == 'y':
                mod2sm = 5.*(Beta*self.xiasm[i]-BB*0.5*self.xiasm[i-nrbin])+A02+A12/r+A22/r**2.
                fo.write(str(self.rl[i])+' '+str(mod2)+' '+str(mod2sm)+'\n')            
            modl.append(mod2)
        if wo == 'y':
            fo.close()      
            

        ChiSq = 0
        chid = 0
        Chioff = 0
        dl = []
        for i in range(0,self.nbin):
            dl.append(self.xim[i]-modl[i])  
        chit = dot(dot(dl,self.invt),dl)
        if v == 'y':
            print(dl,chit)
        BBfac = (log(BB/self.BB)/self.Bp)**2.
        #Btfac = ((Beta-self.B0)/self.Bt)**2.
        Btfac = (log(Beta/self.B0)/self.Bt)**2.
        return chit+BBfac+Btfac


def plot_2dlik(file):
	from matplotlib import pyplot as plt
	chicol=2
	col1name=r'$\alpha_{||}$'
	col2name=r'$\alpha_{\perp}$'

	d = np.loadtxt(file).transpose()
	chi2 = d[chicol]
	prob = np.exp(-0.5*chi2)
	pnorm = np.sum(prob)
	ma1 = np.sum(prob*d[0])/pnorm
	ma2 = np.sum(prob*d[1])/pnorm
	s1 = np.sum(prob*d[0]**2.)/pnorm
	s2 = np.sum(prob*d[1]**2.)/pnorm
	sig1 = np.sqrt(s1-ma1**2.)
	sig2 = np.sqrt(s2-ma2**2.)
	crp = np.sum(prob*d[0]*d[1])/pnorm-ma1*ma2
	w = (chi2-np.min(chi2)) < 1
	indmin = np.argmin(chi2)
	print(col1name+'='+str(ma1)+'+/-'+str(sig1))
	print(col2name+'='+str(ma2)+'+/-'+str(sig2))
	print('correlation is '+str(crp/(sig1*sig2)))

	#plot 1D
	vals1 = np.unique(d[0])
	pv = []
	for v in vals1:
		w = d[0] == v
		pv.append(np.sum(prob[w]))
	pv = np.array(pv)/np.sum(pv)
	plt.plot(vals1,pv)
	plt.xlabel(col1name)
	plt.ylabel('likelihood (integrates to 1)')
	xl = [ma1,ma1]
	yl = [0,np.max(pv)]
	plt.plot(xl,yl,'k-')
	xl = [ma1-sig1,ma1-sig1]
	plt.plot(xl,yl,'k:')
	xl = [ma1+sig1,ma1+sig1]
	plt.plot(xl,yl,'k:')
	plt.show()
	

	#plot 2D
	inds = np.argsort(chi2)
	chi2s = chi2[inds]
	ps = 0
	s1 = 0
	s2 = 0
	s3 = 0
	for i in range(0,len(chi2s)):
		chi = chi2s[i]
		ps += np.exp(-0.5*chi)/pnorm
		if ps > 0.68 and s1 ==0:
			chil1 = chi
			s1 = 1
		if ps > 0.95 and s2 ==0:
			chil2 = chi
			s2 = 1
		if ps > 0.999 and s3 ==0:
			chil3 = chi
			s3 = 1

	w1 = chi2 < chil1
	w2 = chi2 < chil2
	w3 = chi2 < chil3
	ms = 15
	plt.plot(d[0][w3],d[1][w3],'sk',label=r'3$\sigma$',ms=ms)
	plt.plot(d[0][w2],d[1][w2],'sb',label=r'2$\sigma$',ms=ms)
	plt.plot(d[0][w1],d[1][w1],'sr',label=r'1$\sigma$',ms=ms)
	plt.xlim(np.min(d[0]),np.max(d[0]))
	plt.ylim(np.min(d[1]),np.max(d[1]))
	plt.legend()
	plt.xlabel(col1name)
	plt.ylabel(col2name)

	plt.show()



def sigreg_2dEZ(file):
    d = np.loadtxt(file).transpose()
    chi2 = d[-1]
    prob = np.exp(-0.5*chi2)
    pnorm = np.sum(prob)
    mar = np.sum(prob*d[0])/pnorm
    map = np.sum(prob*d[1])/pnorm
    sr = np.sum(prob*d[0]**2.)/pnorm
    sp = np.sum(prob*d[1]**2.)/pnorm
    sigr = sqrt(sr-mar**2.)
    sigp = sqrt(sp-map**2.)
    crp = np.sum(prob*d[0]*d[1])/pnorm-mar*map
    w = (chi2-np.min(chi2)) < 1
    indmin = np.argmin(chi2)
    #print(d[0][indmin],d[1][indmin])
    #print(np.max(abs(d[0][w]-d[0][indmin])),np.max(d[0][indmin]-abs(d[0][w])))
    #print(np.max(abs(d[1][w]-d[1][indmin])),np.max(d[1][indmin]-abs(d[1][w])))
    return mar,sigr,map,sigp,np.min(chi2),crp,crp/(sigr*sigp)
    

def Xism_arat_1C_an(dv,icov,rl,mod,dvb,icovb,rlb,B0=1.,spat=.003,spar=.006,mina=.8,maxa=1.2,nobao='n',Bp=.4,Bt=.4,meth='Powell',fout='',dirout='',Nmock=1000,verbose=False):
    from time import time
    import numpy    
    #from optimize import fmin
    from random import gauss, random
    from scipy.optimize import minimize 
    print('try meth = "Nelder-Mead" if does not work or answer is weird')
    bb = baofit3D_ellFull_1cov(dvb,icovb,mod,rlb,dirout=dirout) #initialize for bias prior
    b = baofit3D_ellFull_1cov(dv,icov,mod,rl,dirout=dirout) #initialize for fitting
    b.B0 = B0
    b.Bt = Bt   
    
    bb.Bp = 100.
    bb.BB = 1.
    bb.B0 = B0
    bb.Bt = 100.
    bb.mkxi()
    t = time()  
    
    B = .01
    chiBmin = 100000
    while B < 4.:
        chiB = bb.chi_templ_alphfXX((B,0,0,0,B,0,0,0))
        if chiB < chiBmin:
            chiBmin = chiB          
            BB = B
            #print(BB,chiBmin)
        B += .01    
    #bb.chi_templ_alphfXX((BB,0,0,0,1.,0,0,0),v='y')    
    print(BB,chiBmin)
    b.BB = BB
    b.B0 = BB       
    b.Bp= Bp
    b.Bt = Bt
    fo = open(dirout+'2Dbaofits/arat'+fout+'1covchi.dat','w')
    fg = open(dirout+'2Dbaofits/arat'+fout+'1covchigrid.dat','w')
    chim = 1000
    nar = int((maxa-mina)/spar)
    nat = int((maxa-mina)/spat)
    grid = numpy.zeros((nar,nat))
    pt = 0
    A0 = 0
    A1 = 0
    A2 = 0
    A02 = 0
    A12 = 0
    A22 = 0
    fac = (Nmock-2.-float(len(dv)))/(Nmock-1.)

    for i in range(0,nar):      
        b.ar = mina+spar*i+spar/2.
        #print(b.ar)
        for j in range(0,nat):
            b.at = mina+spat*j+spat/2.
            b.mkxi()
            inl = (b.B0,b.B0)
            (B,B0) = minimize(b.chi_templ_alphfXX_an,inl,method=meth,options={'disp': False}).x
            chi = b.chi_templ_alphfXX_an((B,B0))*fac
            grid[i][j] = chi
            fo.write(str(b.ar)+' '+str(b.at)+' '+str(chi)+'\n')
            fg.write(str(chi)+' ')
            if chi < chim:
                if verbose:
                    print(b.ar,b.at,chi)    
                chim = chi
                alrm = b.ar
                altm = b.at
                Bm = B
                Betam = B0
        fg.write('\n')
    b.ar = alrm
    b.at = altm 
    b.mkxi()
    b.mkxism()
    chi = b.chi_templ_alphfXX_an((Bm,Betam),wo='y',fw=fout) #writes out best-fit model
    print(alrm,altm,chim)#,alphlk,likm
    alph = (alrm*altm**2.)**(1/3.)
    b.ar = alph
    b.at = alph 
    b.mkxi()
    b.mkxism()
    chi = b.chi_templ_alphfXX_an((Bm,Betam),wo='y',fw=fout+'ep0')
    #print chi
    fo.close()
    fg.close()
    ans = sigreg_2dEZ(dirout+'2Dbaofits/arat'+fout+'1covchi.dat')#sigreg_2dme(dirout+'2Dbaofits/arat'+fout+'1covchi',spar=spar,spat=spat)
    print('<alpha_||>,sigma(||),<alpha_perp>,sigma_perp,min(chi2),cov_||,perp,corr_||,perp:')
    print(ans)
    return True



if __name__ == '__main__':
    import sys
    '''
    example for 1D fit
    '''
    #This is setup to run the data in the exampledata folder, which is BOSS DR11 data
    c1 = np.loadtxt('exampledata/cov0CMASSreconNScomb1DR118st2.dat')
    c2 = np.loadtxt('exampledata/cov0CMASSreconNScomb2DR118st2.dat')
    ct = (c1+c2)/2. #two independent sets of mocks are averaged for the DR11 covariance matrix
    d = np.loadtxt('exampledata/xi0ACbossv1NScombreconbs8st2.dat').transpose()[1]
    r = np.loadtxt('exampledata/xi0ACbossv1NScombreconbs8st2.dat').transpose()[0]
    mod = np.loadtxt('exampledata/xi0camb_Nacc0n3.00.051.9_And.dat').transpose()[1]
    #mod = np.loadtxt('BAOtemplates/xi0Challenge_matterpower0.406.010.015.00.dat').transpose()[1]
    modsm = np.loadtxt('BAOtemplates/xi0smChallenge_matterpower0.406.010.015.00.dat').transpose()[1] #don't see smooth model from back then, shouldn't matter for this example
    spa = .001
    mina = .8
    maxa = 1.2
    cl = doxi_isolike(d,ct,mod,modsm,r,spa=spa,mina=mina,maxa=maxa)
    al = [] #list to be filled with alpha values
    for i in range(0,len(cl)):
        a = .8+spa/2.+spa*i
        al.append(a)
    #below assumes you have matplotlib to plot things, if not, save the above info to a file or something
    from matplotlib import pyplot as plt
    plt.plot(al,cl-min(cl),'k-')
    plt.show()
    
    
    '''
    2D fit from DR12
    '''
    min = 50.
    max = 150. #the minimum and maximum scales to be used in the fit
    maxb = 80. #the maximum scale to be used to set the bias prior
    
    dir = 'exampledata/Ross_2016_COMBINEDDR12/' 
    ft = 'Ross_2016_COMBINEDDR12_'
    zb = 'zbin3_' #change number to change zbin
    binc = 0 #change number to change bin center
    bs = 5. #the bin size
    bc = 'post_recon_bincent'+str(binc)+'.dat' 
    c = np.loadtxt(dir+ft+zb+'covariance_monoquad_'+bc)
    d0 = np.loadtxt(dir+ft+zb+'correlation_function_monopole_'+bc).transpose()[1]
    d2 = np.loadtxt(dir+ft+zb+'correlation_function_quadrupole_'+bc).transpose()[1]
    if len(c) != len(d0)*2:
        print('MISMATCHED data and cov matrix!')
    dv = [] #empty list to become data vector
    dvb = [] #empty list to become data vector for setting bias prior
    rl = [] #empty list to become list of r values to evaluate model at 
    rlb  = [] #empty list to become list of r values to evaluate model at for bias prior
    mini = 0
    for i in range(0,len(d0)):
        r = i*bs+bs/2.+binc
        if r > min and r < max:
            dv.append(d0[i])
            rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
            rl.append(rbc) 
            if mini == 0:
                mini = i #minimum index number to be used for covariance matrix index assignment
            if r < maxb:
                dvb.append(d0[i])
                rlb.append(rbc)
    for i in range(0,len(d2)):
        r = i*bs+bs/2.+binc
        if r > min and r < max:
            dv.append(d2[i])
            rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.)
            rl.append(rbc)
            if r < maxb:
                dvb.append(d2[i])
                rlb.append(rbc)

    dv = np.array(dv)
    print(len(dv))
    covm = zeros((len(dv),len(dv))) #will become covariance matrix to be used with data vector
    #need to cut it to correct size
    for i in range(0,len(c)):
        if i < len(d0):
            ri = i*bs+bs/2.+binc
            indi = i-mini
        else:
            ri = (i-len(d0))*bs+bs/2.+binc
            indi = len(dv)//2+i-mini-len(d0)    
        for j in range(0,len(c)):       
            if j < len(d0):
                rj = j*bs+bs/2.+binc
                indj = j-mini
            else:
                rj = (j-len(d0))*bs+bs/2.+binc
                indj = len(dv)//2+j-mini-len(d0)
            if ri > min and ri < max and rj > min and rj < max:
                #print ri,rj,i,j,indi,indj
                covm[indi][indj] = c[i][j]
    invc = pinv(covm) #the inverse covariance matrix to pass to the code
    covmb = zeros((len(dvb),len(dvb)))
    for i in range(0,len(dvb)):
        if i < len(dvb)//2:
            indi = i
        else:
            indi = i-len(dvb)//2+len(covm)//2
        for j in range(0,len(dvb)):
            if j < len(dvb)//2:
                indj = j
            else:
                indj = j-len(dvb)//2+len(covm)//2
            covmb[i][j] = covm[indi][indj]
    invcb = pinv(covmb)
    mod = 'Challenge_matterpower0.44.02.54.015.01.0.dat' #BAO template used     
    fout = ft+zb+bc
    spa = .001
    mina = .8
    maxa = 1.2
    Xism_arat_1C_an(dv,invc,rl,mod,dvb,invcb,rlb,verbose=True)
