import BAOfit as bf
import numpy as np
import sys

rmin = 50
rmax = 150
maxb = 80.
bs = 5.
binc = 0

datadir = '/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/Reconstruction/Stage1/Multipoles/new_bins/'
data = datadir +sys.argv[1]+'.txt'
d = np.loadtxt(data).transpose()
dv = [] #empty list to become data vector
dvb = [] #empty list to become data vector for setting bias prior
rl = [] #empty list to become list of r values to evaluate model at	
rlb  = [] #empty list to become list of r values to evaluate model at for bias prior
mini = int(rmin//bs)
for i in range(0,len(d[0])):
	r = d[0][i]
	if r > rmin and r < rmax:
		dv.append(d[1][i])
		rbc = r#.75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
		rl.append(rbc) 
		if r < maxb:
			dvb.append(d[1][i])
			rlb.append(rbc)
for i in range(0,len(d[0])):
	r = d[0][i]
	if r > rmin and r < rmax:
		dv.append(d[2][i])
		rbc = r#.75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.)
		rl.append(rbc)
		if r < maxb:
			dvb.append(d[2][i])
			rlb.append(rbc)

dv = np.array(dv)
print(len(dv))
covm = np.zeros((len(dv),len(dv))) #will become covariance matrix to be used with data vector
covf = np.loadtxt('/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/Reconstruction/Stage2/covariances/xi_covariance/cov_matrix_xi-EZmocks-1Gpc_rsd_RecIso_post.txt').transpose()
maxind = 348
for i in range(0,len(covf[0])):
	ind1 = covf[0][i]
	ind2 = covf[1][i]
	if ind1 < maxind and ind2 < maxind:
		if ind1 < maxind//2:
			ic1 = ind1-mini
			r1 = ind1*bs+bs/2.
		else:
			ic1 = len(dv)/2+ind1-maxind//2-mini
			r1 = (ind1-maxind//2)*bs+bs/2.
		if ind2 < maxind//2:
			ic2 = ind2-mini
			r2 = ind2*bs+bs/2.
		else:
			ic2 = len(dv)/2+ind2-maxind//2-mini	
			r2 = (ind2-len(d[0]))*bs+bs/2.
		if r1 > rmin and r2 > rmin and r1 < rmax and r2 < rmax:
			#print(ic1,ic2,r1,r2,covf[2][i])
			covm[int(ic1)][int(ic2)] = covf[2][i]	

invc = np.linalg.pinv(covm) #the inverse covariance matrix to pass to the code
covmb = np.zeros((len(dvb),len(dvb)))
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
invcb = np.linalg.pinv(covmb)
#mod = 'Challenge_matterpower0.5933.058.515.00.dat' #BAO template used		
#fout = 'desi_challeng1_ajr_prerec_0.5933.058.515.00'
mod = 'Challenge_matterpower0.593.04.07.015.00.dat' #BAO template used		
fout = 'desi_challeng2_ajr_'+sys.argv[1]+'_00.593.04.07.015.00.00'

spa = .001
mina = .8
maxa = 1.2
bf.Xism_arat_1C_an(dv,invc,rl,mod,dvb,invcb,rlb,verbose=True)
				