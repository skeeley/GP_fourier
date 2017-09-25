import numpy as np

from . import mkcovs, kron_ops
from . import realfftbasis as rffb


def comp_bases_frs_fourier(x,dims,minlens,nxcirc = None,condthresh = 1e8):
	# Compute least-squares regression sufficient statistics in DFT basis
	# Python version of this NOT complete 9/15/17!
	#
	# [dd,wwnrm,Bfft] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,condthresh)
	#
	# INPUT:
	# -----
	#           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
	#        dims [m x 1] - number of coefficients along each stimulus dimension
	#     minlens [m x 1] - minimum length scale for each dimension (can be scalar)
	#      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
	#  condthresh [1 x 1] - condition number for thresholding for small eigenvalues OPTIONAL
	#
	# OUTPUT:
	# ------
	#     dd (struct) - carries sufficient statistics for linear regresion
	#  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
	#   Bfft  {1 x p} - cell array with DFT bases for each dimension

	# 1e8 is default value (condition number on prior covariance)


	dims = np.array(np.reshape(dims,(1,-1)))
	minlens = np.array(np.reshape(minlens,(1,-1)))

	# Set circular bounardy (for n-point fft) to avoid edge effects, if needed
	if nxcirc is None:
	    #nxcirc = np.ceil(max([dims(:)'+minlens(:)'*4; dims(:)'*1.25]))'
	    nxcirc = np.ceil(np.max(np.concatenate((dims+minlens*4 ,dims*1.25), axis = 0), axis = 0))


	nd = np.size(dims) # number of filter dimensions
	if np.size(minlens) is 1: #% make vector out of minlens, if necessary
	    minlens = np.repmat(minlens,nd,1)


	# Determine number of freqs and make Fourier basis for each dimension
	# cdiagvecs = [None for x in range(nd)] # eigenvalues for each dimension
	# Bfft = [None for x in range(nd)] # Fourier basis matrix for each filter dimension
	# wvecs = [None for x in range(nd)] # Fourier frequencies for each filter dimension
	# ncoeff = np.zeros([nd,1])


	#fprintf('\ncompLSsuffstats_fourier:\n # filter freqs per stimulus dim:');
	# Loop through dimensions
	for jj  in np.arange(nd):
	    #careful here, the mkcov_ASDfactored function uses minlens and 1 as the lensc and rho params
	    wve[jj] = rffb.comp_wvec(nxcirc[jj],minlens[0][jj], condthresh)
	   	Bfft[jj] = rffb.realfftbasis(nx,nxcirc[jj],wvec)[0]

	#fprintf('\n Total # Fourier coeffs represented: %d\n\n', prod(ncoeff));

	def f(switcher):  
	    # switch based on stimulus dimension
	    return{
	    1: #% 1 dimensional stimulus
	        [np.square(2*pi/nxcirc[0]) * np.square(wvecs[0]), # normalized freqs squared
	        ii = true(len(wwnrm),1)] #indices to keep 
	        
	    # case 2, % 2 dimensional stimulus
	        
	    #     % Form full frequency vector and see which to cut
	    #     Cdiag = kron(cdiagvecs{2},cdiagvecs{1});
	    #     ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep 
	                    
	    #     % compute vector of normalized frequencies squared
	    #     [ww1,ww2] = ndgrid(wvecs{1},wvecs{2});
	    #     wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).^2 ...
	    #         (ww2(ii)*(2*pi/nxcirc(2))).^2];
	        
	    # case 3, % 3 dimensional stimulus

	    #     Cdiag = kron(cdiagvecs{3},(kron(cdiagvecs{2},cdiagvecs{1})));
	    #     ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep
	        
	    #     % compute vector of normalized frequencies squared
	    #     [ww1,ww2,ww3] = ndgrid(wvecs{1},wvecs{2},wvecs{3});
	    #     wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).mv ^2, ...
	    #         (ww2(ii)*(2*pi/nxcirc(2))).^2, ....,
	    #         (ww3(ii)*(2*pi/nxcirc(3))).^2];
	        
	    # otherwise
	    #     error('compLSsuffstats_fourier.m : doesn''t yet handle %d dimensional filters\n',nd);
		}[switcher]        
	[wwnrm, ii] = f(nd)

	# Calculate stimulus sufficient stats in Fourier domain
	Bx = kronmulttrp(Bfft,np.transpose(x)) # convert to Fourier domain
	Bx = Bx[ii,:] # prune unneeded freqs


	return wwnrm, Bfft, ii

	# ------ Examine speed of knonmult for DFT operation ----------
	# [nsamps,xwid] = size(x); % Determine size of stimulus and its dimensions
	# % Relative cost of FFT first vs. x'*x first if we did full kronecker
	# (nsamps*nf*(xwid+nf)) > (xwid^2*(nsamps+nf)+nf^2*xwid)
	# % Old / slow way:  
	# xx = x'*x;  xy = x'*y;  then FFT
	# xx = kronmult(Bfft,kronmult(Bfft,xx)')';
	# xy = kronmult(Bfft,xy);
	# -------------------------------------------------------------  


















def compLSsuffstats_fourier(x,y,dims,minlens,nxcirc = None,condthresh = 1e8):
	# Compute least-squares regression sufficient statistics in DFT basis
	# Python version of this NOT complete 9/15/17!
	#
	# [dd,wwnrm,Bfft] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,condthresh)
	#
	# INPUT:
	# -----
	#           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
	#        dims [m x 1] - number of coefficients along each stimulus dimension
	#     minlens [m x 1] - minimum length scale for each dimension (can be scalar)
	#      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
	#  condthresh [1 x 1] - condition number for thresholding for small eigenvalues OPTIONAL
	#
	# OUTPUT:
	# ------
	#     dd (struct) - carries sufficient statistics for linear regresion
	#  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
	#   Bfft  {1 x p} - cell array with DFT bases for each dimension

	# 1e8 is default value (condition number on prior covariance)


	dims = np.array(np.reshape(dims,(1,-1)))
	minlens = np.array(np.reshape(minlens,(1,-1)))

	# Set circular bounardy (for n-point fft) to avoid edge effects, if needed
	if nxcirc is None:
	    #nxcirc = np.ceil(max([dims(:)'+minlens(:)'*4; dims(:)'*1.25]))'
	    nxcirc = np.ceil(np.max(np.concatenate((dims+minlens*4 ,dims*1.25), axis = 0), axis = 0))


	nd = np.size(dims) # number of filter dimensions
	if np.size(minlens) is 1: #% make vector out of minlens, if necessary
	    minlens = np.repmat(minlens,nd,1)


	# Determine number of freqs and make Fourier basis for each dimension
	# cdiagvecs = [None for x in range(nd)] # eigenvalues for each dimension
	# Bfft = [None for x in range(nd)] # Fourier basis matrix for each filter dimension
	# wvecs = [None for x in range(nd)] # Fourier frequencies for each filter dimension
	# ncoeff = np.zeros([nd,1])


	#fprintf('\ncompLSsuffstats_fourier:\n # filter freqs per stimulus dim:');
	# Loop through dimensions
	for jj  in np.arange(nd):
	    #careful here, the mkcov_ASDfactored function uses minlens and 1 as the lensc and rho params
	   	prs = [[minlens[0][jj],1],dims[0][jj]]
	    cdiagvecs[jj],Bfft[jj],wvecs[jj] = mkcovs.mkcov_ASDfactored(prs,nxcirc[jj], condthresh, compfftbasis= 1)
	    ncoeff[jj] = len(cdiagvecs[jj]) # number of coeffs

	#fprintf('\n Total # Fourier coeffs represented: %d\n\n', prod(ncoeff));

	def f(switcher):  
	    # switch based on stimulus dimension
	    return{
	    1: #% 1 dimensional stimulus
	        [np.square(2*pi/nxcirc(1)) * np.square(wvecs[0]), # normalized freqs squared
	        ii = true(len(wwnrm),1)], #indices to keep 
	        
	    # case 2, % 2 dimensional stimulus
	        
	    #     % Form full frequency vector and see which to cut
	    #     Cdiag = kron(cdiagvecs{2},cdiagvecs{1});
	    #     ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep 
	                    
	    #     % compute vector of normalized frequencies squared
	    #     [ww1,ww2] = ndgrid(wvecs{1},wvecs{2});
	    #     wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).^2 ...
	    #         (ww2(ii)*(2*pi/nxcirc(2))).^2];
	        
	    # case 3, % 3 dimensional stimulus

	    #     Cdiag = kron(cdiagvecs{3},(kron(cdiagvecs{2},cdiagvecs{1})));
	    #     ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep
	        
	    #     % compute vector of normalized frequencies squared
	    #     [ww1,ww2,ww3] = ndgrid(wvecs{1},wvecs{2},wvecs{3});
	    #     wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).mv ^2, ...
	    #         (ww2(ii)*(2*pi/nxcirc(2))).^2, ....,
	    #         (ww3(ii)*(2*pi/nxcirc(3))).^2];
	        
	    # otherwise
	    #     error('compLSsuffstats_fourier.m : doesn''t yet handle %d dimensional filters\n',nd);
		}[switcher]        
	[wwnrm, ii] = f(nd)

	# Calculate stimulus sufficient stats in Fourier domain
	Bx = kronmulttrp(Bfft,np.transpose(x)) # convert to Fourier domain
	Bx = Bx[ii,:] # prune unneeded freqs

	dd = {}
	dd['xx'] = np.matmul(Bx,Bx, transpose_b=True)
	dd['xy'] = Bx * y

	# Fill in other statistics
	dd['yy'] = np.transpose(y)*y # marginal response variance
	dd['nsamps'] = np.size(y,1)  # total number of samples

	return dd, wwnrm, Bfft, i 

	# ------ Examine speed of knonmult for DFT operation ----------
	# [nsamps,xwid] = size(x); % Determine size of stimulus and its dimensions
	# % Relative cost of FFT first vs. x'*x first if we did full kronecker
	# (nsamps*nf*(xwid+nf)) > (xwid^2*(nsamps+nf)+nf^2*xwid)
	# % Old / slow way:  
	# xx = x'*x;  xy = x'*y;  then FFT
	# xx = kronmult(Bfft,kronmult(Bfft,xx)')';
	# xy = kronmult(Bfft,xy);
	# -------------------------------------------------------------  



