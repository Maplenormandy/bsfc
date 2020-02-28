''' Get pos vector for different spectral fits in BSFC 

branchB: MOD1-3
branchA: MOD4



'''


import MDSplus
import cPickle as pkl
from bsfc_moment_fitter import get_hirexsr_lam_bounds
import numpy as np
import matplotlib.pyplot as plt

tht=1
primary_line = 'w'
primary_impurity = 'Ar'
branchB=True

shot=1101014030

specTree = MDSplus.Tree('spectroscopy', shot)

if branchB:
    # pos vectors for detector modules 1-3
    pos1 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:POS').data()
    pos2 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD2:POS').data()
    pos3 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD3:POS').data()

    # wavelengths for each module
    lam1 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:LAMBDA').data()
    lam2 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD2:LAMBDA').data()
    lam3 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD3:LAMBDA').data()

    pos_tot = np.hstack([pos1,pos2,pos3])
    lam_tot = np.hstack([lam1,lam2,lam3])
else:
    # 1 detector module
    pos_tot = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD4:POS').data()

    # wavelength
    lam_tot = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD4:LAMBDA').data()


branchNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS{:s}.{:s}LIKE'.format(str(tht) if tht!=0 else '','HE' if branchB else 'H'))

# mapping from pixels to chords (wavelength x space pixels, but wavelength axis is just padding)
chmap = branchNode.getNode('BINNING:CHMAP').data()
pixels_to_chords = chmap[0,:]

# -----
# find over which wavelengths the pos vector should be averaged at every time
# get lambda bounds for specific BSFC line fit
lam_bounds = get_hirexsr_lam_bounds(primary_impurity, primary_line)

lam_all = branchNode.getNode('SPEC:LAM').data()

# exclude empty chords
mask = lam_all[0,0,:]!=-1
lam_masked = lam_all[:,:,mask]

# lambda vector does not change over time, so just use tbin=0
tbin=0
w0=[]; w1=[]
for chbin in np.arange(lam_masked.shape[2]):
    bb = np.searchsorted(lam_masked[:,tbin,chbin], lam_bounds)
    w0.append(bb[0])
    w1.append(bb[1])
# -----

# get time vector
#tmp=np.asarray(branchNode.getNode('SPEC:SIG').dim_of(1))
#mask = tmp>-1
#time = tmp[mask]
        
# form chords
pos_ave = []
for chord in np.arange(lam_masked.shape[2]):
    pos_ave.append( np.mean(pos_tot[w0[chord]:w1[chord], pixels_to_chords == chord,:], axis=(0,1) ))
pos_ave = np.array(pos_ave)




# =========================================

# compare to w wavelength case
with open('/home/sciortino/bsfc/helpers/bsfc_hirex_%d_%s_%s.pkl'%(shot,primary_line, primary_impurity), 'rb') as f:
    data = pkl.load(f)

pos_true = data['pos']

# chmap.shape = (195,487) ---> wavelength x space pixels
# lam_all.shape = (195, 417, 32) ---> 417 times
# len(w0) = 32 ---> indices of arrays of length 195
# pos4.shape = (195, 487, 4) -----> wavelength x space pixels 
# pos_true.shape = (32,4)


fig,ax = plt.subplots(2,2)
axx = ax.flatten()
for i in [0,1,2,3]:
    pcm = axx[i].pcolormesh(pos_tot[:,:,i].T)
    axx[i].axis('equal')
    fig.colorbar(pcm, ax=axx[i])


fig,ax = plt.subplots()
plt.plot(pos_ave)

axx = ax.flatten()
for i in [0,1,2,3]:
    pcm = axx[i].pcolormesh(pos_[:,:,i].T)
    axx[i].axis('equal')
    fig.colorbar(pcm, ax=axx[i])
