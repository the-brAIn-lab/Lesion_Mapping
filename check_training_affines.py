import nibabel as nib
import glob
import numpy as np

affines = []
for f in glob.glob('/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training_Split/Masks/*.nii.gz'):
    affines.append(nib.load(f).affine)
affines = np.array(affines)
print('Unique training mask affines:', len(np.unique(affines, axis=0)))
