'''
do BAO fits for various cases
directories will get hardcoded to where AJR stores things
'''

import numpy as np
from BAOfit import *
from matplotlib import pyplot as plt

def DESItest(samp='full'):
	'''
	2D fit from DR12
	'''
	rmin = 50.
	rmax = 140. #the minimum and maximum scales to be used in the fit
	rmaxb = 70. #the maximum scale to be used to set the bias prior
	binc = 0
	dir = '/Users/ashleyross/Dropbox/DESI/mockxi_fromSA/' 
	bs = 4. #the bin size
	c = np.loadtxt(dir+'ELG-rsd-z1z2-0.70-1.10-SDECALS-'+samp+'-mock.cov').transpose()
	xi02 = np.loadtxt(dir+'ELG-rsd-z1z2-0.70-1.10-SDECALS-'+samp+'-meanmock.xi02').transpose()
	d0 = xi02[1]
	d2 = xi02[2]
	dv = [] #empty list to become data vector
	dvb = [] #empty list to become data vector for setting bias prior
	rl = [] #empty list to become list of r values to evaluate model at	
	rlb  = [] #empty list to become list of r values to evaluate model at for bias prior
	mini = 0
	for i in range(0,len(d0)):
		r = i*bs+bs/2.+binc
		if r > rmin and r < rmax:
			dv.append(d0[i])
			rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
			rl.append(rbc) 
			if mini == 0:
				mini = i #minimum index number to be used for covariance matrix index assignment
			if r < rmaxb:
				dvb.append(d0[i])
				rlb.append(rbc)
	for i in range(0,len(d2)):
		r = i*bs+bs/2.+binc
		if r > rmin and r < rmax:
			dv.append(d2[i])
			rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.)
			rl.append(rbc)
			if r < rmaxb:
				dvb.append(d2[i])
				rlb.append(rbc)

	dv = np.array(dv)
	#print(len(dv))
	print(rl)
	covm = np.zeros((len(dv),len(dv))) #will become covariance matrix to be used with data vector
	corrm = np.zeros((len(dv),len(dv)))
	#need to cut it to correct size
	for i in range(0,len(c[0])):
		ri = c[2][i]
		rj = c[3][i]
		if ri > rmin and ri < rmax and rj > rmin and rj < rmax:
			cov = c[-1][i]
			ii = c[0][i]
			jj = c[1][i]
			#if ii < len(d0) and jj > len(d0):
			#	cov = 0
			#if ii > len(d0) and jj < len(d0):
			#	cov = 0
			#if ii == jj:
				#cov = 0.01/ri**2.
			#	cov = c[-1][i]
			#else:
				#cov = 0.001/(ri*rj)
			#	cov = 0

			if ii < len(d0):
				indi = ii -mini
			else:
				indi = len(dv)//2+ii-len(d0)-mini
			if jj < len(d0):
				indj = jj -mini
			else:
				indj = len(dv)//2+jj-len(d0)-mini
			#print(indi,indj,cov,i,ii,jj)	
			indi = int(indi)
			indj = int(indj)

			covm[indi][indj] = cov
	for i in range(0,len(covm)):
		for j in range(0,len(covm)):
			corrm[i][j] = covm[i][j]/np.sqrt(covm[i][i]*covm[j][j])		
			
	#print(covm)
	#print(np.min(covm),np.max(covm))
	plt.imshow(corrm)
	plt.show()
	invc = np.linalg.pinv(covm)#,rcond=1.e-5) #the inverse covariance matrix to pass to the code
	plt.imshow(invc)
	plt.show()

	covmb = np.zeros((len(dvb),len(dvb)))
	for i in range(0,len(c[0])):
		ri = c[2][i]
		rj = c[3][i]
		if ri > rmin and ri < rmaxb and rj > rmin and rj < rmaxb:
			cov = c[-1][i]

			ii = c[0][i]
			jj = c[1][i]
			#if ii == jj:
			#	cov = c[-1][i]
			#else:
			#	cov = 0
			#	cov = 0.001/(ri*rj)

			if ii < len(d0):
				indi = ii -mini
			else:
				indi = len(dvb)//2+ii-len(d0)-mini
			if jj < len(d0):
				indj = jj -mini
			else:
				indj = len(dvb)//2+jj-len(d0)-mini
			indi = int(indi)
			indj = int(indj)
			print(indi,indj,cov,i,ii,jj,ri,rj)

			covmb[indi][indj] = cov

	invcb = np.linalg.pinv(covmb)#,rcond=1.e-5)
	mod = 'Challenge_matterpower0.353.06.010.015.00.dat' #BAO template used		
	fout = 'samp'
	spa = .001
	mina = .8
	maxa = 1.2
	Xism_arat_1C_an(dv,invc,rl,mod,dvb,invcb,rlb,dirout=dir,fout='fout',verbose=True)

if __name__ == '__main__':
	DESItest()