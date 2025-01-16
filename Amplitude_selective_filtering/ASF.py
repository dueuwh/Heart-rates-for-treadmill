import numpy as np

def amplitudeSelectiveFiltering(C_rgb, amax = 0.002, delta = 0.0001):
    '''
    Input: Raw RGB signals with dimensions 3xL, where the R channel is column 0
    Output: 
    C = Filtered RGB-signals with added global mean, 
    raw = Filtered RGB signals
    '''

    L = C_rgb.shape[1]
    C = (1/(np.mean(C_rgb,1)))


    #line 1
    C = np.transpose(np.array([C,]*(L)))* C_rgb -1
    #line 2       
    F = abs(np.fft.fft(C,n=L,axis=1)/L) #L -> C_rgb.shape[0]

    #line 3   
    W = (delta / np.abs(F[0,:])) #F[0,:]  is the R-channel
    
    #line 4
    W[np.abs(F[0,:]<amax)] = 1
    W = W.reshape([1,L])

    #line 5
    Ff = np.multiply(F,(np.tile(W,[3,1])))
    
    #line 6
    C = np.transpose(np.array([(np.mean(C_rgb,1)),]*(L))) * np.abs(np.fft.ifft(Ff)+1)
    raw = np.abs(np.fft.ifft(Ff)+1)
    return C.T, raw.T
