import BAOfit as bf
import numpy as np

rmin = 50
rmax = 150
maxb = 80.
bs = 5.

data = '/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/UNIT/xi_1Gpc/xil_rsd.txt'
d = np.loadtxt(data).transpose()
dv = [] #empty list to become data vector
dvb = [] #empty list to become data vector for setting bias prior
rl = [] #empty list to become list of r values to evaluate model at	
rlb  = [] #empty list to become list of r values to evaluate model at for bias prior
mini = 0
for i in range(0,len(d[0])):
	r = i*bs+bs/2.+binc
	if r > min and r < max:
		dv.append(d[1][i])
		rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
		rl.append(rbc) 
		if mini == 0:
			mini = i #minimum index number to be used for covariance matrix index assignment
		if r < maxb:
			dvb.append(d[1][i])
			rlb.append(rbc)
for i in range(0,len(d[0])):
	r = i*bs+bs/2.+binc
	if r > min and r < max:
		dv.append(d[2][i])
		rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.)
		rl.append(rbc)
		if r < maxb:
			dvb.append(d[2][i])
			rlb.append(rbc)

dv = np.array(dv)
print(len(dv))
covm = zeros((len(dv),len(dv))) #will become covariance matrix to be used with data vector
covf = np.loadtxt('/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/EZmocks/xi_EZ_mocks/covariance/cov_matrix_rsd.txt').transpose()
for i in range(0,len(covf[0])):
	ind1 = covf[0][i]
	ind2 = covf[1][i]
	if ind1 and ind2 < len(d[0])*2:
		if ind1 < len(d[0]):
			ic1 = ind1-mini
		else:
			ic1 = len(dv)/2+ind1-len(d[0])-mini
		if ind2 < len(d[0]):
			ic2 = ind2-mini
		else:
			ic2 = len(dv)/2+ind2-len(d[0])-mini	
		covm[ic1][ic2] = covf[2][i]	

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
mod = 'Challenge_matterpower0.5933.058.515.00.dat' #BAO template used		
fout = ft+zb+bc
spa = .001
mina = .8
maxa = 1.2
Xism_arat_1C_an(dv,invc,rl,mod,dvb,invcb,rlb,verbose=True)
				