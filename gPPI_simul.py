#!/usr/bin/env python3
""" BSM/PPI simulation for Figure 8

Dependencies: numpy, scipy, pandas, joblib, nilearn.
Both estimators receive identical BOLD signals within each simulated subject.

Example commands used for the main, HRF, and ITI analyses:

python gPPI_fig8_github.py \
  --hrf variable \
  --delta 0 0.05 0.1 0.2 0.4 \
  --bold-snr 0.5 1 1.5 2 \
  --latent-snr 0.1 0.25 0.5 1 \
  --source-effect 0.1 0.25 0.5 0.75 \
  --baseline-coupling 0 0.2 0.4 0.6 \
  --mean-iti 4.5 \
  --reps 1000 \
  --jobs 196 \
  --batch-size 10 \
  --condition-chunk 8 \
  --output fig8_aligned_main

python gPPI_fig8_github.py \
  --hrf both \
  --delta 0 0.05 \
  --bold-snr 0.5 1 1.5 2 \
  --latent-snr 0.1 0.25 0.5 1 \
  --source-effect 0.5 \
  --baseline-coupling 0.2 \
  --mean-iti 4.5 \
  --reps 100 \
  --jobs 256 \
  --batch-size 10 \
  --condition-chunk 8 \
  --output fig8_aligned_hrf2

python gPPI_fig8_github.py \
  --hrf variable \
  --delta 0 0.1 \
  --bold-snr 0.5 1 1.5 2 \
  --latent-snr 0.1 0.25 0.5 1 \
  --source-effect 0.5 \
  --baseline-coupling 0.2 \
  --mean-iti 1.5 3 4.5 6 \
  --reps 1000 \
  --jobs 196 \
  --batch-size 10 \
  --condition-chunk 8 \
  --output fig8_aligned_iti

"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import argparse
import itertools
import time
import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.stats import gamma
from joblib import Parallel, delayed
from nilearn.glm.first_level import compute_regressor


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def custom_spm_hrf(tr, oversampling=10, time_length=32.0, delay=6.0):
    # Use the same HRF normalization for data generation and PPI reconvolution.
    dt = tr / oversampling
    t = np.linspace(0, time_length, int(np.rint(time_length / dt)))
    h = gamma.pdf(t, delay, loc=dt, scale=1.0) - 0.167 * gamma.pdf(t, 16.0, loc=dt, scale=1.0)
    return h / h.sum()


def make_design(rng, n_trials, mean_iti, trial_duration, tr):
    itis = np.maximum(0.5, rng.exponential(mean_iti, size=n_trials))
    onsets = np.cumsum(itis + trial_duration) - trial_duration
    durations = np.full(n_trials, trial_duration)
    duration = np.ceil((onsets[-1] + trial_duration + 20.0) / tr) * tr
    frames = np.arange(0, duration, tr)
    return onsets, durations, frames


def lss_operator(onsets, durations, frames, method="original"):
    n = len(onsets)
    full, _ = compute_regressor((onsets, durations, np.ones(n)), 'spm', frames)
    full = full[:, 0]
    intercept = np.ones(len(frames))
    op = np.empty((n, len(frames)))
    for i in range(n):
        reg, _ = compute_regressor((onsets[i:i+1], durations[i:i+1], np.ones(1)), 'spm', frames)
        target = reg[:, 0]
        x = np.column_stack((target, full - target, intercept))
        if method == "original":
            # Retain the LSS operator used for the reported BSM simulations.
            op[i] = (linalg.pinv(x.T @ x) @ x.T)[0]
        elif method == "direct":
            # used this for some testing only.
            op[i] = linalg.pinv(x)[0]
        else:
            raise ValueError(f"Unknown LSS method: {method}")
    return op


def dct_basis(m, n_basis):
    """Tried to match the DCT basis method for deconvolution."""
    row = np.arange(m, dtype=float)[:, None]
    col = np.arange(1, n_basis, dtype=float)[None, :]
    basis = np.empty((m, n_basis), dtype=float)
    basis[:, 0] = 1.0 / np.sqrt(m)
    basis[:, 1:] = np.sqrt(2.0 / m) * np.cos(np.pi * (2.0 * row + 1.0) * col / (2.0 * m))
    return basis


def deconv_operator(n_scans, tr, microtime, alpha):
    """Precompute the DCTforward model and ridge for PPI."""
    n = n_scans
    pad = 128
    n_basis = int(np.ceil(n / 4))  # Restrict the neural basis to low-frequency components.
    basis = dct_basis(n * microtime + pad, n_basis)
    h = custom_spm_hrf(tr, oversampling=microtime, delay=6)
    indices = np.arange(n) * microtime + pad
    hxb = np.empty((n, n_basis), dtype=float)
    for k in range(n_basis):
        hxb[:, k] = np.convolve(basis[:, k], h, mode='full')[indices]
    xb = basis[pad:, :]
    gram = hxb.T @ hxb + alpha * np.eye(n_basis)
    factor = linalg.cho_factor(gram, check_finite=False)
    return xb, hxb, factor


def ar1_noise(rng, n, rho):
    ''' make sure both methods use the same rho.... '''
    # Original recursive AR(1) process, normalized to sample SD 1.
    innovations = rng.standard_normal(n)
    from scipy.signal import lfilter
    noise = lfilter([1.0], [1.0, -rho], innovations)
    return zscore(noise)


def t_reject(betas, alpha=0.05):
    t, p = stats.ttest_1samp(betas, 0.0)
    return float(p < alpha), float(t)


def compute_bsm(y_src, y_tgt, mod, op):
    s = zscore(op @ y_src)
    t = zscore(op @ y_tgt)
    m = zscore(mod)
    x = np.column_stack((s, m, zscore(s*m), np.ones(len(m))))
    return float(linalg.lstsq(x, t)[0][2])


def compute_ppi(y_src, y_tgt, mod, onsets, durations, frames, microframes, tr, microtime, operator):
    xb, hxb, factor = operator
    # Estimate a smooth neural source from BOLD using ridge-regularized DCT inversion.
    weights = linalg.cho_solve(factor, hxb.T @ y_src, check_finite=False)
    neural = zscore(xb @ weights)
    psych, _ = compute_regressor((onsets, durations, mod), 'spm', frames)
    # Multiply the deconvolved source by the un-convolved psychological boxcar.
    psych_micro, _ = compute_regressor((onsets, durations, mod), None, microframes)
    product = neural * zscore(psych_micro[:, 0])
    h = custom_spm_hrf(tr, oversampling=microtime, delay=6)
    # Sample the reconvolved interaction at the same frame times as observed BOLD.
    ppi = np.convolve(product, h, mode='full')[::microtime][:len(frames)]
    x = np.column_stack((zscore(y_src), zscore(psych[:, 0]), zscore(ppi), np.ones(len(frames))))
    return float(linalg.lstsq(x, zscore(y_tgt))[0][2])


def simulate_subject(rng, params, cfg, design, op, deconv):
    onsets, durations, frames, microframes = design
    n = len(onsets)
    m = zscore(rng.standard_normal(n))
    alpha = params['source_effect']
    gamma_c = params['baseline_coupling']
    latent_snr = params['latent_snr']
    bold_snr = params['bold_snr']
    delta = params['delta']
    source_signal = alpha * m
    source_latent = source_signal + zscore(rng.standard_normal(n)) * (source_signal.std() / latent_snr)
    target_signal = gamma_c * source_latent + delta * (source_latent * m) + alpha * m
    target_latent = target_signal + zscore(rng.standard_normal(n)) * (target_signal.std() / latent_snr)
    if params['hrf'] == 'canonical':
        src_delay = tgt_delay = 6.0
    else:
        src_delay, tgt_delay = rng.uniform(6 - cfg['hrf_delay_range'], 6 + cfg['hrf_delay_range'], size=2)
    def src_hrf(tr_arg, oversampling=50):
        return custom_spm_hrf(tr_arg, oversampling=oversampling, delay=src_delay)
    def tgt_hrf(tr_arg, oversampling=50):
        return custom_spm_hrf(tr_arg, oversampling=oversampling, delay=tgt_delay)
    src, _ = compute_regressor((onsets, durations, source_latent), src_hrf, frames)
    tgt, _ = compute_regressor((onsets, durations, target_latent), tgt_hrf, frames)
    src = src[:, 0]
    tgt = tgt[:, 0]
    rho = params['bold_ar1']
    y_src = src + ar1_noise(rng, len(src), rho) * (src.std() / bold_snr)
    y_tgt = tgt + ar1_noise(rng, len(tgt), rho) * (tgt.std() / bold_snr)
    bsm = compute_bsm(y_src, y_tgt, m, op)
    ppi = compute_ppi(y_src, y_tgt, m, onsets, durations, frames, microframes, cfg['tr'], cfg['microtime'], deconv)
    return bsm, ppi


def run_batch(params, cfg, rep_indices, master_seed):
    # Hold the event schedule fixed within each condition and share BOLD data across methods.
    # Repetition seeds are independent of job scheduling and batch size.
    # Design is fixed across batches of the same parameter condition.
    design_rng = np.random.default_rng(np.random.SeedSequence([master_seed, params['condition_id'], 999]))
    onsets, durations, frames = make_design(design_rng, cfg['n_trials'], params['mean_iti'], cfg['trial_duration'], cfg['tr'])
    microframes = np.arange(len(frames) * cfg['microtime']) * cfg['tr'] / cfg['microtime']
    realized_itis = np.diff(np.r_[0.0, onsets]) - np.r_[0.0, durations[:-1]]
    realized_mean_iti = float(np.mean(realized_itis))
    design = onsets, durations, frames, microframes
    op = lss_operator(onsets, durations, frames, cfg["lss_method"])
    deconv = deconv_operator(len(frames), cfg['tr'], cfg['microtime'], cfg['ridge_alpha'])
    rows = []
    for rep in rep_indices:
        rng = np.random.default_rng(np.random.SeedSequence([master_seed, params['condition_id'], int(rep)]))
        bsm_betas = []
        ppi_betas = []
        for _ in range(cfg['n_subjects']):
            b, p = simulate_subject(rng, params, cfg, design, op, deconv)
            bsm_betas.append(b)
            ppi_betas.append(p)
        bsm_reject, bsm_t = t_reject(bsm_betas)
        ppi_reject, ppi_t = t_reject(ppi_betas)
        rows.append({**params, 'rep': int(rep), 'bsm_reject': bsm_reject,
                     'ppi_reject': ppi_reject, 'bsm_t': bsm_t, 'ppi_t': ppi_t,
                     'bsm_mean_beta': float(np.mean(bsm_betas)),
                     'ppi_mean_beta': float(np.mean(ppi_betas)),
                     'realized_mean_iti': realized_mean_iti})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--condition-chunk', type=int, default=8,
                        help='Number of parameter conditions processed and checkpointed together.')
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--n-subjects', type=int, default=38)
    parser.add_argument('--n-trials', type=int, default=408)
    parser.add_argument('--microtime', type=int, default=10)
    parser.add_argument('--ridge-alpha', type=float, default=0.05)
    parser.add_argument('--lss-method', choices=['original', 'direct'], default='original',
                        help='original: pinv(X.T @ X) @ X.T; direct: pinv(X) (diagnostic only)')
    parser.add_argument('--hrf', choices=['both','canonical','variable'], default='variable')
    parser.add_argument('--delta', type=float, nargs='+', default=[0, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument('--bold-snr', type=float, nargs='+', default=[0.5, 1, 1.5, 2])
    parser.add_argument('--latent-snr', type=float, nargs='+', default=[0.1, 0.25, 0.5, 1])
    parser.add_argument('--mean-iti', type=float, nargs='+', default=[4.5])
    parser.add_argument('--source-effect', type=float, nargs='+', default=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument('--baseline-coupling', type=float, nargs='+', default=[0, 0.2, 0.4, 0.6])
    parser.add_argument('--bold-ar1', type=float, nargs='+', default=[0.17])
    parser.add_argument('--output', default='figure8_bsm_ppi')
    parser.add_argument('--resume', action='store_true', help='Skip checkpointed condition chunks.')
    args = parser.parse_args()
    if min(args.reps, args.jobs, args.batch_size, args.condition_chunk) < 1:
        parser.error('reps, jobs, batch-size and condition-chunk must be positive')
    if args.n_subjects < 2 or args.n_trials < 3:
        parser.error('At least 2 subjects and 3 trials required')
    if any(x <= 0 for x in args.mean_iti + args.bold_snr + args.latent_snr):
        parser.error('ITI and SNR must be positive')
    if any(abs(x) >= 1 for x in args.bold_ar1):
        parser.error('AR(1) coefficients must have magnitude below 1')
    cfg = dict(tr=2.0, n_trials=args.n_trials, n_subjects=args.n_subjects,
               trial_duration=1.0, hrf_delay_range=2.0,
               microtime=args.microtime, ridge_alpha=args.ridge_alpha,
               lss_method=args.lss_method)
    hrfs = ['canonical','variable'] if args.hrf == 'both' else [args.hrf]
    factors = itertools.product(hrfs, args.delta, args.bold_snr, args.latent_snr,
                                args.mean_iti, args.source_effect, args.baseline_coupling,
                                args.bold_ar1)
    conditions = [dict(condition_id=i, hrf=h, delta=d, bold_snr=b,
                       latent_snr=l, mean_iti=iti, source_effect=a,
                       baseline_coupling=g, bold_ar1=rho)
                  for i, (h,d,b,l,iti,a,g,rho) in enumerate(factors)]
    from pathlib import Path
    import json
    import hashlib
    outdir = Path(args.output + '_chunks')
    outdir.mkdir(exist_ok=True, parents=True)
    manifest = dict(seed=args.seed, reps=args.reps, n_subjects=args.n_subjects,
                    n_trials=args.n_trials, microtime=args.microtime,
                    ridge_alpha=args.ridge_alpha, lss_method=args.lss_method,
                    tr=cfg['tr'], trial_duration=cfg['trial_duration'],
                    hrf_delay_range=cfg['hrf_delay_range'], conditions=conditions)
    manifest_text = json.dumps(manifest, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_text.encode()).hexdigest()
    manifest_path = outdir / 'manifest.json'
    if args.resume and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get('hash') != manifest_hash:
            parser.error('Existing checkpoint manifest does not match these parameters; choose a new --output')
    elif any(outdir.glob('chunk_*.csv')):
        parser.error('Output directory already has checkpoints; use --resume or a new --output')
    manifest_path.write_text(json.dumps({'hash':manifest_hash, 'config':manifest}, indent=2))
    print(f'{len(conditions)} conditions x {args.reps} repetitions; '
          f'{len(conditions)*args.reps:,} experiments, 2 methods each', flush=True)
    t0 = time.time()
    for chunk_start in range(0, len(conditions), args.condition_chunk):
        chunk = conditions[chunk_start:chunk_start + args.condition_chunk]
        path = outdir / f'chunk_{chunk_start:06d}.csv'
        if args.resume and path.exists():
            print(f'Skipping {path.name}', flush=True)
            continue
        jobs = [(p, np.arange(st, min(st + args.batch_size, args.reps)))
                for p in chunk for st in range(0, args.reps, args.batch_size)]
        pieces = Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(run_batch)(p, cfg, indices, args.seed) for p, indices in jobs)
        frame = pd.DataFrame([row for piece in pieces for row in piece])
        temp = path.with_suffix('.tmp')
        frame.to_csv(temp, index=False)
        temp.replace(path)
        print(f'Checkpointed conditions {chunk[0]["condition_id"]}-'
              f'{chunk[-1]["condition_id"]}; elapsed {(time.time()-t0)/60:.1f} min', flush=True)
    raw_path = Path(args.output + '_repetitions.csv')
    summaries = []
    with raw_path.open('w') as output_handle:
        for i, path in enumerate(sorted(outdir.glob('chunk_*.csv'))):
            df = pd.read_csv(path)
            df.to_csv(output_handle, index=False, header=(i == 0))
            cols = ['condition_id','hrf','delta','bold_snr','latent_snr','mean_iti',
                    'source_effect','baseline_coupling','bold_ar1']
            summaries.append(df.groupby(cols, as_index=False).agg(
                n_reps=('rep','size'), bsm_rate=('bsm_reject','mean'),
                ppi_rate=('ppi_reject','mean'), bsm_mean_t=('bsm_t','mean'),
                ppi_mean_t=('ppi_t','mean'), bsm_mean_beta=('bsm_mean_beta','mean'),
                ppi_mean_beta=('ppi_mean_beta','mean'),
                realized_mean_iti=('realized_mean_iti','mean')))
    summary = pd.concat(summaries, ignore_index=True)
    for method in ('bsm','ppi'):
        rate = summary[f'{method}_rate']
        summary[f'{method}_rate_se'] = np.sqrt(rate*(1-rate)/summary.n_reps)
    summary['paired_rate_difference'] = summary.bsm_rate - summary.ppi_rate
    summary['rate_type'] = np.where(summary.delta == 0, 'false_positive_rate', 'power')
    summary['n_subjects'] = args.n_subjects
    summary['n_trials'] = args.n_trials
    summary['ridge_alpha'] = args.ridge_alpha
    summary['microtime'] = args.microtime
    summary['lss_method'] = args.lss_method
    summary.to_csv(args.output + '_summary.csv', index=False)
    print(f'Finished in {(time.time()-t0)/60:.1f} minutes; '
          f'saved {raw_path} and {args.output}_summary.csv', flush=True)


if __name__ == '__main__':
    main()
