"""
gPPI simulation

Simulation logic:
-----------------
1. Generate randomized event timing
2. Simulate latent neural source activity
3. Generate interaction-driven target activity
4. Convolve neural activity with HRF
5. Add AR(1) noise at controlled SNR
6. Deconvolve source BOLD to estimate neural activity
7. Build canonical Friston-style gPPI regressor
8. Fit gPPI model
9. Estimate recovery / power across simulations
"""

import os
# Limit threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import multiprocessing
import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.stats import zscore, gamma
from joblib import Parallel, delayed
from nilearn.glm.first_level import compute_regressor


############################################################
# setup params
############################################################
### possible parameters
TR = [1, 1.5, 2, 3]
N_TRIALS = [50, 150, 300] # fixed number of trials
trial_duration = [0.25, 0.5, 1, 2]
SOURCE_EFFECT_TRUE = [0.1, 0.25, 0.5, 0.75] 
BASELINE_COUPLING_TRUE = [0.0, 0.2, 0.4, 0.6]
INTERACTION_TRUE_GRID = [0.0, 0.05, 0.1, 0.2, 0.4]
BOLD_SNR_GRID = [0.25, 0.5, 1.0, 2.0] 
LATENT_SNR_GRID = [0.1, 0.25, 0.5, 1.0] 
N_SUBJECTS = [15, 25, 40]
# exponential ITIs
MEAN_ITI = [1, 2, 4, 6]
# random subject-level HRF variability
HRF_DELAY_RANGE = 2 # pluse minus peak of 6
### end of possible parameters

N_REPS = 2 #number of simulations
N_JOBS = max(1, multiprocessing.cpu_count() // 2)
OUTPUT_FILE = "gppi_simulation_results.csv"
DECONV_ALPHA = 0.05
MICROTIME_RESOLUTION = 10 # for 1 second TR this is about 100 hz


############################################################
# custom hrf function. This is from nilearn SPM
############################################################
def custom_spm_hrf( t_r, oversampling=50, time_length=32.0, onset=0.0, delay=6, undershoot=16.0, dispersion=1.0, u_dispersion=1.0, ratio=0.167):

    dt = t_r / oversampling
    time_stamps = np.linspace( 0, time_length, np.rint(float(time_length) / dt).astype(int), )
    time_stamps -= onset
    peak_gamma = gamma.pdf( time_stamps, delay / dispersion, loc=dt, scale=dispersion, )
    undershoot_gamma = gamma.pdf( time_stamps, undershoot / u_dispersion, loc=dt, scale=u_dispersion, )
    hrf = peak_gamma - ratio * undershoot_gamma
    hrf /= hrf.sum()

    return hrf


############################################################
# functions for deconvolution
# these are modified from
# Masharipov, R., Knyazeva, I., Korotkov, A., Cherednichenko, D., & Kireev, M. (2024). Comparison of whole-brain task-modulated functional connectivity methods for fMRI task connectomics. Communications biology, 7(1), 1402.
# see public code at
#  https://github.com/IHB-IBR-department/TMFC_simulations/blob/main/deconvolution/python/bold_deconvolution.py
############################################################
def dctmtx_numpy_vect(N: int, K: int) -> np.ndarray:
    n = np.arange(N)
    C = np.zeros((N, K))
    C[:, 0] = 1 / np.sqrt(N)
    k = np.arange(1, K)
    C[:, 1:K] = (
        np.sqrt(2 / N)
        *
        np.cos(np.pi * (2 * n[:, np.newaxis]) * k / (2 * N))
    )

    return C


def compute_xb_Hxb( N, NT, TR, ):
    # latent res setup
    # NT = number of microtime bins per TR.
    # Example:
    # TR = 2 s
    # NT = 16
    # -> dt = 125 ms
    #
    # This follows the classical SPM logic where neural
    # activity is represented at finer temporal resolution
    # than the observed BOLD sampling rate.

    dt = TR / NT
    k = np.arange(0, N * NT, NT)

    # Canonical HRF at microtime resolution    
    # note that
    # - simulation generation uses subject-specific HRFs
    # - deconvolution assumes canonical HRF

    hrf = custom_spm_hrf( TR, oversampling=NT, delay=6, )

    # Build discrete cosine basis set
    #
    # Instead of directly inverting the BOLD signal,
    # neural activity is represented as a weighted
    # combination of smooth cosine basis functions.
    # Otherwise it is ill posed

    M = N * NT + 128
    xb = dctmtx_numpy_vect(M, N)

    # Convolve each basis function with HRF
    # Hxb becomes the forward model relating latent neural
    # basis coefficients to observed BOLD signal.

    Hxb = np.zeros((N, N))

    for i in range(N):
        Hx = np.convolve( xb[:, i], hrf, mode='full', )
        Hxb[:, i] = Hx[k + 128]

    xb = xb[128:, :]

    return xb, Hxb


# timeseries deconvolution
def ridge_regress_deconvolution( BOLD, alpha, xb, Hxb, ):

    # Ridge regression
    #
    # Solve:
    # beta = (H'H + alpha*I)^-1 H'Y
    # where:
    #
    # H = convolved basis set
    # Y = observed BOLD signal

    C = np.linalg.solve(
        Hxb.T @ Hxb + alpha * np.eye(Hxb.shape[1]),
        Hxb.T @ BOLD,
    )

    # Reconstruct neural signal in microtime resolution
    neuro = xb @ C
    return neuro.flatten()


############################################################
# BUILD DESIGN
############################################################

def build_design_matrices( tr, n_trials, trial_duration_value, mean_iti, ):

    ############################################################
    # BUILD TIMING
    ############################################################
    itis = np.random.exponential(scale=mean_iti, size=n_trials)
    # should we set minimum ITI?
    itis = np.maximum(0.5, itis)

    # creat timing
    all_onsets = []
    current_time = 0.0

    for iti in itis:
        current_time += iti
        all_onsets.append(current_time)
        current_time += trial_duration_value

    all_onsets = np.array(all_onsets)
    all_durations = np.ones(n_trials) * trial_duration_value
    run_duration = all_onsets[-1] + trial_duration_value + 20 # pad a bit at end

    # round up to nearest TR
    run_duration = np.ceil(run_duration / tr) * tr

    # frame times for nilearn
    frame_times = np.arange(0, run_duration, tr)

    # Precompute deconv basis set
    # This is reused across all subjects in a condition,
    xb, Hxb = compute_xb_Hxb( N=len(frame_times), NT=MICROTIME_RESOLUTION, TR=tr)

    # latent frame times
    microtime_frame_times = np.arange( 0, run_duration, tr / MICROTIME_RESOLUTION)

    return {
        "all_onsets": all_onsets,
        "all_durations": all_durations,
        "frame_times": frame_times,
        "microtime_frame_times": microtime_frame_times,
        "xb": xb,
        "Hxb": Hxb,
    }


############################################################
# simulation
############################################################
def run_single_rep(
    interaction_true,
    bold_snr,
    latent_snr,
    n_trials,
    source_effect_true,
    baseline_coupling_true,
    n_subjects,
    design,
):

    all_onsets = design["all_onsets"]
    all_durations = design["all_durations"]
    frame_times = design["frame_times"]
    microtime_frame_times = design["microtime_frame_times"]
    xb = design["xb"] #this is latent activity basis set in (N_latet timepoints, N_basis_functions)
    Hxb = design["Hxb"] # the above convolved with HRF

    ########################################################
    # Store subject-level gPPI effects
    ########################################################

    # Each subject contributes ONE interaction beta
    # These subject-level betas will later be tested
    # against zero at the group level.

    interaction_betas = []
    main_betas = []
    mod_betas = []

    for subj_idx in range(n_subjects):

        ########################################################
        # subject-level hrf variability
        ########################################################
        src_delay = np.random.uniform(6.0 - HRF_DELAY_RANGE, 6.0 + HRF_DELAY_RANGE)
        tgt_delay = np.random.uniform(6.0 - HRF_DELAY_RANGE, 6.0 + HRF_DELAY_RANGE)

        #hrf functions for compute regressor
        def src_hrf(tr, oversampling=50):
            return custom_spm_hrf(
                tr,
                oversampling=oversampling,
                delay=src_delay,
            )

        def tgt_hrf(tr, oversampling=50):
            return custom_spm_hrf(
                tr,
                oversampling=oversampling,
                delay=tgt_delay,
            )

        ########################################################
        # Generate latent activity
        ########################################################
        '''
        Dial so the laten signal achieve the latent SNR of
        y = signal + noise * (std(signal) / latent SNR)
        '''
        # create random model variables,
        # wonder if we should read from real data?
        moderator = zscore(np.random.randn(n_trials))
        moderator_signal = source_effect_true * moderator

        source_noise = zscore(np.random.randn(n_trials))

        source_latent = (moderator_signal #this alone is the co activation effect
            + source_noise * (np.std(moderator_signal) / (latent_snr + 1e-12))) # add noise to source to achieve desired latent SNR
        
        ### now construct the target signal
        interaction_drive = source_latent * moderator

        target_signal_component = (
            baseline_coupling_true * source_latent   # intrinsic coupling
            + interaction_true * interaction_drive      # interaction effect
            + source_effect_true * moderator    # co-activation effect
        )

        target_noise = zscore(np.random.randn(n_trials))
        target_latent = (target_signal_component + target_noise * (np.std(target_signal_component) / (latent_snr + 1e-12)))

        ########################################################
        # HRF convolution
        ########################################################
        src_reg, _ = compute_regressor(
            (all_onsets, all_durations, source_latent),
            hrf_model=src_hrf,
            frame_times=frame_times,
        )

        tgt_reg, _ = compute_regressor(
            (all_onsets, all_durations, target_latent),
            hrf_model=tgt_hrf,
            frame_times=frame_times,
        )

        psych_reg, _ = compute_regressor(
            (all_onsets, all_durations, moderator),
            hrf_model="spm",
            frame_times=frame_times,
        ) #this is task regressor

        ########################################################
        # Psychological regressor at latent resolution
        # Classical gPPI computes interaction at neural resolution before reconvolution.
        ########################################################
        psych_micro_reg, _ = compute_regressor(
            (all_onsets, all_durations, moderator),
            hrf_model=None,
            frame_times=microtime_frame_times,
        )

        src_signal = src_reg[:, 0]
        tgt_signal = tgt_reg[:, 0]
        psych_signal = psych_reg[:, 0]
        psych_micro_signal = psych_micro_reg[:, 0]

        # Add AR(1) noise

        rho = 0.5
        src_noise = np.random.randn(len(src_signal))
        tgt_noise = np.random.randn(len(tgt_signal))

        for i in range(1, len(src_noise)):
            src_noise[i] += rho * src_noise[i - 1]
            tgt_noise[i] += rho * tgt_noise[i - 1]

        src_noise = zscore(src_noise)
        tgt_noise = zscore(tgt_noise)

        # scale SNR
        y_src = src_signal + src_noise * ( np.std(src_signal) / (bold_snr + 1e-12))
        y_tgt = tgt_signal + tgt_noise * ( np.std(tgt_signal) / (bold_snr + 1e-12))

        ########################################################
        # Deconvolution
        # Estimate latent neural activity from observed BOLD
        # using ridge-regularized basis regression.
        # Output neural signal exists in latent resolution.
        ########################################################
        source_neural_est = ridge_regress_deconvolution(
            y_src,
            alpha=DECONV_ALPHA,
            xb=xb,
            Hxb=Hxb,
        )

        ########################################################
        # Build PPI-style neural interaction
        ########################################################
        source_neural_est = zscore(source_neural_est)
        psych_micro_signal = zscore(psych_micro_signal)
        interaction_neural = (source_neural_est * psych_micro_signal )

        ########################################################
        # Reconvolution
        ########################################################
        canonical_hrf = custom_spm_hrf(
            TR,
            oversampling=MICROTIME_RESOLUTION,
            delay=6,
        )

        ppi_signal_full = np.convolve(
            interaction_neural,
            canonical_hrf,
            mode='full',
        )

        # Downsample back to TR resolution
        ppi_signal = ppi_signal_full[ ::MICROTIME_RESOLUTION ][:len(frame_times)]

        ########################################################
        # Build gPPI model
        ########################################################
        X = np.column_stack([
            zscore(y_src),
            zscore(psych_signal),
            zscore(ppi_signal),
            np.ones(len(frame_times)),
        ])

        # Subject-level regression
        #
        # beta_hat:
        #
        # [0] = source main effect (baseline coupling)
        # [1] = moderator main effect (task effect)
        # [2] = interaction effect (gPPI)
        # [3] = intercept

        beta_hat, _, _, _ = linalg.lstsq(X, zscore(y_tgt))

        interaction_betas.append(beta_hat[2])
        main_betas.append(beta_hat[0])
        mod_betas.append(beta_hat[1])

    # Group-level statistics
    interaction_betas = np.array(interaction_betas)
    main_betas = np.array(main_betas)
    mod_betas = np.array(mod_betas)

    interaction_t, interaction_p = stats.ttest_1samp( interaction_betas, popmean=0.0, )
    main_t, main_p = stats.ttest_1samp( main_betas, popmean=0.0, )
    mod_t, mod_p = stats.ttest_1samp( mod_betas, popmean=0.0, )

    return {
        "interaction_reject": float(interaction_p < 0.05),
        "interaction_mean_beta": np.mean(interaction_betas),
        "interaction_t": interaction_t,

        "main_reject": float(main_p < 0.05),
        "main_mean_beta": np.mean(main_betas),
        "main_t": main_t,

        "mod_reject": float(mod_p < 0.05),
        "mod_mean_beta": np.mean(mod_betas),
        "mod_t": mod_t,
    }


############################################################
# run condition
############################################################
def run_condition(
    interaction_true,
    bold_snr,
    latent_snr,
    tr,
    n_trials,
    trial_duration_value,
    source_effect_true,
    baseline_coupling_true,
    n_subjects,
    mean_iti,
):

    design = build_design_matrices( tr, n_trials, trial_duration_value, mean_iti, )

    rep_results = []
    for rep in range(N_REPS):

        result = run_single_rep(
            interaction_true,
            bold_snr,
            latent_snr,
            n_trials,
            source_effect_true,
            baseline_coupling_true,
            n_subjects,
            design,
        )

        rep_results.append(result)

    rep_df = pd.DataFrame(rep_results)

    return {
        "interaction_true": interaction_true,
        "bold_snr": bold_snr,
        "latent_snr": latent_snr,
        "tr": tr,
        "n_trials": n_trials,
        "trial_duration": trial_duration_value,
        "source_effect_true": source_effect_true,
        "baseline_coupling_true": baseline_coupling_true,
        "n_subjects": n_subjects,
        "mean_iti": mean_iti,

        # interaction effect
        "interaction_power": rep_df["interaction_reject"].mean(),
        "interaction_mean_beta": rep_df["interaction_mean_beta"].mean(),
        "interaction_t": rep_df["interaction_t"].mean(),

        # source main effect
        "main_power": rep_df["main_reject"].mean(),
        "main_mean_beta": rep_df["main_mean_beta"].mean(),
        "main_t": rep_df["main_t"].mean(),

        # moderator main effect
        "mod_power": rep_df["mod_reject"].mean(),
        "mod_mean_beta": rep_df["mod_mean_beta"].mean(),
        "mod_t": rep_df["mod_t"].mean(),
    }


############################################################
# build jobs
############################################################
simulation_jobs = [
    (
        interaction_true,
        bold_snr,
        latent_snr,
        tr,
        n_trials,
        trial_duration_value,
        source_effect_true,
        baseline_coupling_true,
        n_subjects,
        mean_iti,
    )

    for interaction_true in INTERACTION_TRUE_GRID
    for bold_snr in BOLD_SNR_GRID
    for latent_snr in LATENT_SNR_GRID
    for tr in [TR]
    for n_trials in [N_TRIALS]
    for trial_duration_value in [trial_duration]
    for source_effect_true in SOURCE_EFFECT_TRUE
    for baseline_coupling_true in BASELINE_COUPLING_TRUE
    for n_subjects in [N_SUBJECTS]
    for mean_iti in [MEAN_ITI]
]


############################################################
# run simulations
############################################################
import time

t0 = time.time()

results = Parallel(
    n_jobs=N_JOBS,
    batch_size=4,
)(
    delayed(run_condition)(
        interaction_true,
        bold_snr,
        latent_snr,
        tr,
        n_trials,
        trial_duration_value,
        source_effect_true,
        baseline_coupling_true,
        n_subjects,
        mean_iti,
    )

    for (
        interaction_true,
        bold_snr,
        latent_snr,
        tr,
        n_trials,
        trial_duration_value,
        source_effect_true,
        baseline_coupling_true,
        n_subjects,
        mean_iti,
    ) in simulation_jobs
)

runtime = time.time() - t0

print("\nBenchmark completed")
print(f"Runtime: {runtime:.2f} seconds")

results_df = pd.DataFrame(results)
results_df.to_csv( OUTPUT_FILE, index=False, )