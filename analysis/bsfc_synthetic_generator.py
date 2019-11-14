# -*- coding: utf-8 -*-
"""
Tests bsfc_main versus synthetic data, to see the quality of the fits

@author: normandy
"""

import readline
import MDSplus

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import scipy.stats

speedOfLight = 2.998e+5 # speed of light in km/s

# %%

class SyntheticGenerator:
    def __init__(self, shot, tht, branchB, dataLine, tbin, tinst=0.5):
        # Do some branching to load the right node
        ana = 'ANALYSIS'
        if tht>0:
            ana += str(tht)
        if branchB:
            ana += '.HELIKE'
        else:
            ana += '.HLIKE'


        self.tbin = tbin

        specTree = MDSplus.Tree('spectroscopy', shot)
        profileNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.'+ana+'.PROFILES.'+dataLine)

        # Load wavelength, and inverted radial locations + profiles
        self.rho = profileNode.getNode('rho').data()[tbin,:]
        self.pro = profileNode.getNode('pro').data()[:,tbin,:]

        # Load the voxel matrices
        self.voxel = profileNode.getNode('CONFIG:VOXEL').data()[tbin,:len(self.rho),:]
        self.velvoxel = profileNode.getNode('CONFIG:VELVOXEL').data()[:,tbin,:len(self.rho),:]
        self.tinst = tinst

    def calculateTrueMoments(self, mf, chbin):
        # Get the normalized line width w
        w2 = (self.pro[3,:]+self.tinst) * mf.lines.lam[0]**2 / mf.lines.m_kev[0]
        # (Approximate) the line-of-sight velocity
        v = self.pro[1,:] * mf.lines.lam[0] / speedOfLight
        emiss = self.pro[0,:]

        # Calculate average moments by using equations (2.6-2.8)
        m0 = np.dot(emiss, self.voxel)
        m1 = np.dot(emiss*v, self.voxel)*1e3
        m2_0 = np.dot(emiss*w2, self.voxel)*1e6

        vmean = m1*1e-3 / m0
        vdiff = (v[np.newaxis,:] - vmean[:,np.newaxis])**2
        m2_1 = np.diag(np.dot(vdiff*emiss[np.newaxis,:], self.voxel))*1e6

        m2 = m2_0 + m2_1

        return np.array([m0[chbin], m1[chbin], m2[chbin]])

    def calculateTrueMeasurements(self, mf, chbin):
        moms = self.calculateTrueMoments(mf, chbin)
        m0_true = moms[0]
        v_true = moms[1]/moms[0]/1e3*speedOfLight/mf.lines.lam[0]
        ti_true = moms[2]/moms[0]/1e6*mf.lines.m_kev[0]/mf.lines.lam[0]**2

        return np.array([m0_true, v_true, ti_true])

    def generateSyntheticSpectrum(self, mf, chbin):
        mf.fitSingleBin(self.tbin, chbin, nsteps=1, n_hermite=3)
        countsMultiplier = np.zeros(len(mf.lines.lam))
        bf = mf.fits[self.tbin][chbin]

        if not bf.good:
            return

        for j in range(len(mf.lines.lam)):
            measurement_ml = bf.lineModel.modelMeasurements(bf.theta_ml, line=j)
            countsMultiplier[j] = measurement_ml[0]
        #countsMultiplier[1] = countsMultiplier[0]*0.1
        print countsMultiplier

        noise, center, scale, herm = bf.lineModel.unpackTheta(bf.theta_ml)
        noise = noise[0]

        countsMultiplier = countsMultiplier / countsMultiplier[0]

        # Get the normalized line width w
        w2 = (self.pro[3,:]+self.tinst) * mf.lines.lam[0]**2 / mf.lines.m_kev[0]
        # (Approximate) the line-of-sight velocity
        v = self.pro[1,:] * mf.lines.lam[0] / speedOfLight
        emiss = self.pro[0,:]

        lam = mf.lam_all[:,self.tbin,chbin]

        lamEdge = np.zeros(len(lam)+1)
        lamEdge[1:-1] = (lam[1:] + lam[:-1]) / 2
        lamEdge[-1] = 2 * lamEdge[-2] - lamEdge[-3]
        lamEdge[0] = 2 * lamEdge[1] - lamEdge[2]
        lamEdge = lamEdge
        dlam = np.mean(np.diff(lam))

        specBr = np.zeros(lam.shape)
        voxelWeight = self.voxel[:,chbin]
        whitefield = mf.whitefield[:,self.tbin,chbin]

        for j in range(len(self.rho)):
            for l in range(len(mf.lines.lam)):
                pline = scipy.stats.norm.cdf(lamEdge, loc=v[j]+mf.lines.lam[l], scale=np.sqrt(w2[j])*mf.lines.sqrt_m_ratio[l])
                line = np.diff(pline)/dlam * countsMultiplier[l] * emiss[j] * voxelWeight[j]
                specBr += line

        specBr += noise

        specBrSamp = np.random.poisson(lam=specBr*whitefield)/whitefield
        specBrSampSig = np.sqrt(specBrSamp*whitefield)/whitefield


        mf.specBr_all[:,self.tbin,chbin] = specBrSamp
        mf.sig_all[:,self.tbin,chbin] = specBrSampSig
