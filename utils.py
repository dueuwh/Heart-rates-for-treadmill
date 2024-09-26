# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:34:08 2024

@author: ys
"""

import numpy as np
import math as m
from scipy.signal import butter, filtfilt, welch
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
from numba import cuda, njit, prange
import torch

def bbox2_CPU(img):
    """
    Args:
        img (ndarray): ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    nzrows = np.nonzero(rows)
    nzcols = np.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(float32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power

class BPM:
    """
    Provides BPMs estimate from BVP signals using CPU.

    BVP signal must be a float32 numpy.ndarray with shape [num_estimators, num_frames].
    """
    def __init__(self, data, fps, startTime=0, minHz=0.65, maxHz=4., verb=False):
        """
        Input 'data' is a BVP signal defined as a float32 Numpy.ndarray with shape [num_estimators, num_frames]
        """
        self.nFFT = 2048//1  # freq. resolution for STFTs
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz


    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # -- BPM estimate
        #Normalized Power에서 획득하는 SNR은, 일반 SNR과 비교하면 min(Power) 값이 penalty term 역할을 함.
        Pmax = np.argmax(Power, axis=1)  # power max
        SNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        Power = (Power-np.min(Power))/(np.max(Power)-np.min(Power))
        pSNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        return Pfreqs[Pmax.squeeze()], SNR, pSNR, Pfreqs, Power

def cpu_OMIT(signal):
    """
    OMIT method on CPU using Numpy.

    Álvarez Casado, C., Bordallo López, M. (2022). Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).
    """
    X = signal
    Q, R = np.linalg.qr(X)
    S = Q[:, 0].reshape(1, -1)
    P = np.identity(3) - np.matmul(S.T, S)
    Y = np.dot(P, X)
    bvp= Y[1, :]
    return bvp

def cpu_LGI(signal):
    """
    LGI method on CPU using Numpy.

    Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
    """
    X = signal
    U, _, _ = np.linalg.svd(X)
    S = U[:, :]
    S = np.expand_dims(S, 2)
    sst = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - sst
    Y = np.matmul(P, X)
    bvp = Y[:, 1, :]
    return bvp

class Gaussian():
    def __init__(self, mean, stdd):
        self.mean = mean
        self.stdd = stdd
    
    def update(self, mean, stdd):
        self.mean = mean
        self.stdd = stdd
    
    def get(self, x):
        return (1/(2*m.pi*((self.stdd)**2))**1/2)*m.exp((-(x-self.mean)**2)/(2*(self.stdd)**2))

def signal_filtering(signal, low_band=round(5/6, 4), high_band=3.0, fs=30, N=2):
    #미국 기준 HR 30~240 구간 측정 가능해야함. fs : 카메라 FPS
    [b_pulse, a_pulse] = butter(N, [low_band / fs * 2, high_band / fs * 2], btype='bandpass')
    rst_signal_0 = filtfilt(b_pulse, a_pulse, np.double(signal[:, 0]))
    rst_signal_1 = filtfilt(b_pulse, a_pulse, np.double(signal[:, 1]))
    rst_signal_2 = filtfilt(b_pulse, a_pulse, np.double(signal[:, 2]))

    return np.concatenate((np.expand_dims(rst_signal_0, axis=1), np.expand_dims(rst_signal_1, axis=1), np.expand_dims(rst_signal_2, axis=1)), axis=1)

class SkinExtractionConvexHull:
    """
        This class performs skin extraction on CPU/GPU using a Convex Hull segmentation obtained from facial landmarks.
    """
    def __init__(self,device='CPU'):
        """
        Args:
            device (str): This class can execute code on 'CPU' or 'GPU'.
        """
        self.device = device
    
    def extract_skin(self,image, ldmks):
        """
        This method extract the skin from an image using Convex Hull segmentation.

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].

        Returns:
            Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        from pyVHR.extraction.sig_processing import MagicLandmarks
        aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]        
        # face_mask convex hull 
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
        mask = np.expand_dims(mask,axis=0).T

        # left eye convex hull
        left_eye_ldmks = ldmks[MagicLandmarks.left_eye]
        aviable_ldmks = left_eye_ldmks[left_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            left_eye_mask = np.array(img)
            left_eye_mask = np.expand_dims(left_eye_mask,axis=0).T
        else:
            left_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # right eye convex hull
        right_eye_ldmks = ldmks[MagicLandmarks.right_eye]
        aviable_ldmks = right_eye_ldmks[right_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            right_eye_mask = np.array(img)
            right_eye_mask = np.expand_dims(right_eye_mask,axis=0).T
        else:
            right_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # mounth convex hull
        mounth_ldmks = ldmks[MagicLandmarks.mounth]
        aviable_ldmks = mounth_ldmks[mounth_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            mounth_mask = np.array(img)
            mounth_mask = np.expand_dims(mounth_mask,axis=0).T
        else:
            mounth_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # apply masks and crop 
        if self.device == 'GPU':
            image = cupy.asarray(image)
            mask = cupy.asarray(mask)
            left_eye_mask = cupy.asarray(left_eye_mask)
            right_eye_mask = cupy.asarray(right_eye_mask)
            mounth_mask = cupy.asarray(mounth_mask)
        skin_image = image * mask * (1-left_eye_mask) * (1-right_eye_mask) * (1-mounth_mask)

        if self.device == 'GPU':
            rmin, rmax, cmin, cmax = bbox2_GPU(skin_image)
        else:
            rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)

        cropped_skin_im = skin_image
        if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
            cropped_skin_im = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]

        if self.device == 'GPU':
            cropped_skin_im = cupy.asnumpy(cropped_skin_im)
            skin_image = cupy.asnumpy(skin_image)

        return cropped_skin_im, skin_image


def holistic_mean(im, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [1,3], where 1 is the single estimator,
        and 3 are r-mean, g-mean and b-mean.
    """
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
            if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH)
                    or (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b 
    return mean

if __name__ == "__main__":
    passroun

