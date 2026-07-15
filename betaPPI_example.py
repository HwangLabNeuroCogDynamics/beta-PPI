"""
beta-PPI example, using seed-based FC approach. 
This example demonstrates a voxelwise beta-series psychophysiological
interaction (beta-PPI) analysis.

1. Load a beta-series NIfTI image (one volume per trial).
2. Load a seed ROI mask.
3. Extract the trial-wise beta series from the seed.
4. Fit a beta-PPI model at every voxel:

       voxel_beta ~ seed_beta * moderator

5. Save the interaction coefficient as a NIfTI image.

# note you need nilearn and statsmodel:

# this was tested was these versions
nilearn: 0.12.1
statsmodels: 0.14.6
nibabel:5.3.3 

"""

from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nilearn.maskers import NiftiMasker
from scipy.stats import zscore

BETA_SERIES = #input of your single trial beta series, can be from 3dLSS or GLMSingle
SEED_MASK = # the mask of your source ROI
BRAIN_MASK = #whole brain ROI or whatever taret ROI you want to restrict to
TRIAL_TABLE = # a dataframe that has a trial wise moderator
MODERATOR_COL = "moderator" 
OUTPUT_NII = Path("betaPPI_interaction_beta.nii.gz")


# Load data
beta_img = nib.load(BETA_SERIES)
trial_df = pd.read_csv(TRIAL_TABLE)
moderator = trial_df[MODERATOR_COL].to_numpy()

# Extract seed beta series
seed_masker = NiftiMasker(mask_img=SEED_MASK, standardize=False)

# Shape should be in (n_trials, n_seed_voxels)
seed_data = seed_masker.fit_transform(beta_img)
seed_beta = seed_data.mean(axis=1)

# Extract whole-brain beta series
brain_masker = NiftiMasker(mask_img=BRAIN_MASK, standardize=False)

# in this (n_trials, n_voxels)
brain_data = brain_masker.fit_transform(beta_img)
seed_beta = zscore(seed_beta, nan_policy="omit")
moderator = zscore(moderator, nan_policy="omit")
interaction = seed_beta * moderator

X = np.column_stack( [ np.ones(len(seed_beta)), seed_beta, moderator, interaction]) #need to have intercept, source and evoke effects

# Fit voxelwise beta-PPI note here we use joblib to do parallel computing otherwise it takes forever
interaction_beta = np.full(brain_data.shape[1], np.nan)

from joblib import Parallel, delayed

# Fit voxelwise beta-PPI
def fit_voxel(y):
    """Fit beta-PPI model for a single voxel."""

    if np.isnan(y).any():
        return (np.nan,) * 6

    y = zscore(y, nan_policy="omit")

    # skip constant voxels
    if np.isnan(y).any() or np.std(y) == 0:
        return (np.nan,) * 6

    fit = sm.OLS(y, X).fit()
    
        fit.params[1] # seed beta (intrinsic FC)
        fit.tvalues[1],# seed tval
        fit.params[3], #interaction beta
        fit.tvalues[3],#interaction tval
        fit.pvalues[1], # p val
        fit.pvalues[3], 
    )

NJOBs = 24 #24 coores
results = Parallel( n_jobs=NJOBs, verbose=5, )( delayed(fit_voxel)(brain_data[:, voxel]) for voxel in range(brain_data.shape[1]))
results = np.asarray(results)
seed_beta_map = results[:, 0]
seed_t_map = results[:, 1]
interaction_beta_map = results[:, 2]
interaction_t_map = results[:, 3]

# Save result
maps = {
    "SeedBeta": seed_beta_map,
    "SeedT": seed_t_map,
    "InteractionBeta": interaction_beta_map,
    "InteractionT": interaction_t_map,
}

for name, values in maps.items():
    img = brain_masker.inverse_transform(values)
    img.to_filename(f"betaPPI_{name}.nii.gz")

