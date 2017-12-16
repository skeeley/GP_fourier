import numpy as np
import warnings

from . import fft_ops as rffb
import tensorflow as tf

def mkcovdiag_ASD(len_sc,rho,nxcirc,wvec= None,wwnrm= None, calc_derivs = None):
#  Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
# 
#  [cdiag,dcdiag,ddcdiag] = mkcovdiag_ASD(rho,l,nxcirc,wvecsq)
# 
#  Compute discrete ASD (RBF kernel) eigenspectrum using frequencies in [0, nxcirc].
#  See mkCov_ASD_factored for more info
# 
#  INPUT (all python 1d lists!):
#          len - length scale of ASD kernel (determines smoothness)
#          rho - maximal prior variance ("overall scale")
#       nxcirc - number of coefficients to consider for circular boundary 
#         wvecsq - vector of squared frequencies for DFT 
#         
#  OUTPUT:
#      cdiag [nxcirc x 1] - vector of eigenvalues of C for frequencies in w
#     dcinv [nxcirc x 2] - 1st derivs [dC^-1/drho, dC^-1/dlen]
#        dc [nxcirc x 2] - 1st derivs [dC / drho , dC / dlen]
#     ddcinv [nxcirc x 3] - 2nd derivs of C-1 w.r.t [drho^2, drho*dlen, dlen^2]
# 
# Note: nxcirc = nx corresponds to having a circular boundary 


# Compute diagonal of ASD covariance matrix
	if wvec is not None:
		wvecsq = np.square(wvec)
		const = np.square(2*np.pi/nxcirc) # constant 
		ww = wvecsq*const  # effective frequency vector
	elif wwnrm is not None:
		ww = wwnrm
	else:
		print("please provide either wvec or a normalized wvec into this function")




	cdiag = np.sqrt(2*np.pi)*rho*len_sc*np.exp(-.5*ww*np.square(len_sc))

	# 1st derivative of inv(Cdiag)
	if calc_derivs is not None:
	####NOT WORKING YET!!!	(shouldn't really need this if using tensorflow...)

	    nw = np.size(wvecsq)
	    dcinv = [(-1/len_sc + len_sc*ww)/cdiag,-(1/rho)/cdiag] # dC^-1/drho


	# 1st derivative of Cdiag 
	    dc = [(1/len_sc - len_sc*ww)*cdiag[:,1, 1], (1./rho)*np.ones(nw,1)] # dC/drho
	         # dC/dl


	# 2nd derivative of inv(Cdiag)
	    ddcinv = ([(2/np.square(len_sc) - ww + np.square(len_sc)*np.square(ww)), # d^2 C^-1 /dl^2
	        (1/rho)*(1/len_sc - len_sc*ww),   # d^2 C^-1 /drho dl
	        (2/np.square(rho))*np.ones(nw,1)]     # d^2 C^-1 /drho^2
	        / cdiag[:,np.ones(1,3)])

	        


	    return cdiag, dcinv, dc, ddcinv

	elif calc_derivs is None:
		return cdiag




def mkcov_ASDfactored(prs,nx,nxcirc=None,condthresh = 1e8,compfftbasis = None):
# % Factored representation of ASD covariance matrix in Fourier domain
# %
# % [Cdiag,U,wvec] = mkcov_ASDfactored(prs,nx,opts)
# %
# % Covariance represented as C = U*sdiag*U'
# % where U is unitary (in some larger basis) and sdiag is diagonal
# %
# %  C_ij = rho*exp(((i-j)^2/(2*l^2))
# %
# % INPUT:
# % ------
# %   prs [2 x 1]  - ASD parameters [len_sc = length scale; rho - maximal variance; ]:
# %    nx [1 x 1]  - number of regression coeffs
# % 
# % Note: nxcirc = nx gives circular boundary
# %
# % OUTPUT:
# % -------
# %   cdiag [ni x 1] - vector with thresholded eigenvalues of C
# %       U [ni x nxcirc] - column vectors define orthogonal basis for C (on Reals)
# %    wvec [nxcirc x 1] - vector of Fourier frequencies
# %      ii [nxcirc x 1] - binary vector indicating which frequencies included in U, sdiag

	len_sc = prs[0]
	rho = prs[1]

	# % Parse inputs
	if nxcirc is None:
		nxcirc = nx + np.ceil(4*len_sc) # extends support by 4 stdevs of ASD kernel width



	# % Check that nxcirc isn't bigger than nx
	if nxcirc < nx:
	    warnings.warn('mkcov_ASDfactored: nxcirc < nx. Some columns of x will be ignored')


	# % compute vector of Fourier frequencies
	# maxfreq = np.floor(nxcirc/(np.pi*len_sc)*np.sqrt(.5*np.log(condthresh))) # max
	# if maxfreq < nxcirc/2:
	#     wvec = np.concatenate(([np.arange(int(maxfreq))],[np.arange(-int(maxfreq),0)]),axis = 1)
	#else:
	    # % in case cutoff is above max number of frequencies
	wvec = rffb.comp_wvec(nxcirc, len_sc, condthresh)



	# % Compute diagonal in Fourier domain
	cdiag = mkcovdiag_ASD(len_sc,rho,nxcirc,wvec = wvec) # compute diagonal and Fourier freqs

	# % Compute real-valued discrete Fourier basis U
	if compfftbasis is not None:
	    U = rffb.realfftbasis(nx,nxcirc,wvec)[0]
	    return cdiag, U, wvec
	else:
		return cdiag



def mkcovdiag_ASD_tf(len_sc,rho,nxcirc,wvec= None,wwnrm= None):
#  Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
# 
#  Identical to mkcovdiag_ASD except using tensorflow operations.
#  No need for computing of derivatives here


# Compute diagonal of ASD covariance matrix
	if wvec is not None:
		wvecsq = np.square(wvec)
		const = np.square(2*np.pi/nxcirc) # constant 
		ww = wvecsq*const  # effective frequency vector
	elif wwnrm is not None:
		ww = wwnrm
	else:
		print("please provide either wvec or a normalized wvec into this function")

	cdiag = tf.sqrt(2*np.pi)*rho*len_sc*tf.exp(-.5*ww*tf.square(len_sc))

	
	return cdiag


def mkcovdiag_ASD_wellcond(len_sc,rho,nxcirc,addition = 1e-8, wvec= None,wwnrm= None):

	cdiag = mkcovdiag_ASD(len_sc,rho,nxcirc,wvec,wwnrm)

	return cdiag + addition

def mkcovdiag_ASD_wellcond_tf(len_sc,rho,nxcirc,addition = 1e-8, wvec= None,wwnrm= None):
#  Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
# 
#  Identical to mkcovdiag_ASD except using tensorflow operations.
#  No need for computing of derivatives here


# Compute diagonal of ASD covariance matrix
	if wvec is not None:
		wvecsq = np.square(wvec)
		const = np.square(2*np.pi/nxcirc) # constant 
		ww = wvecsq*const  # effective frequency vector
	elif wwnrm is not None:
		ww = wwnrm
	else:
		print("please provide either wvec or a normalized wvec into this function")

	cdiag = tf.sqrt(2*np.pi)*rho*len_sc*tf.exp(-.5*ww*tf.square(len_sc)) + addition

	
	return cdiag



def mkcovdiag_ASD_nD_tf(len_sc,rho,nxcirc, condthresh, wvec= None,wwnrm= None):
#  Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
# 
#  Identical to mkcovdiag_ASD except using tensorflow operations.
#  No need for computing of derivatives here
# Compute diagonal of ASD covariance matrix.
# threshold again based on length scale!!!
	if wvec is not None:
		wvecsq = np.square(wvec)
		const = np.square(2*np.pi/nxcirc) # constant 
		ww = wvecsq*const  # effective frequency vector
	elif wwnrm is not None:
		ww = wwnrm
	else:
		print("please provide either wvec or a normalized wvec into this function")


	ii = tf.less(ww*np.square(len_sc),2*np.log(condthresh)) # frequency coeffs to keep
	ni = tf.reduce_sum(tf.cast(ii, tf.float32)) # rank of covariance after pruning

	masked_ww = tf.boolean_mask(ww, ii)
	masked_ww = tf.cast(masked_ww, dtype='float32')
	print(masked_ww, len_sc)
	cdiag = tf.sqrt(2*np.pi)*rho*len_sc*tf.exp(-.5*masked_ww*tf.square(len_sc))

	
	return cdiag, ii, ni

	
