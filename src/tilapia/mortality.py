"""Exploratory seven-dose 96-hour LC50 analysis from original cumulative counts.

The original seven-dose experiment has three tanks per dose and 50 fish per
tank. All original-batch doses, including 5 ppt, supply the primary analysis.
Post-selection 18-ppt validation is a different experiment and is rejected
as a source population. The six-dose subset is an explicitly named sensitivity.
"""
from __future__ import annotations
from .runtime import source_member, iter_sources, source_provenance, source_size, source_exists, source_label, source_text, copy_source

from itertools import product
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import chi2
import statsmodels
import statsmodels.api as sm

from .runtime import read_table, write_table, write_json, sha256

INPUTS_BY_TASK = {'mortality': ['salinity_original_seven_group_records']}
TIMES = [0, 24, 48, 72, 96, 120]
DOSES = [0, 5, 10, 15, 20, 25, 30]
SIX_DOSE_SENSITIVITY = [0, 10, 15, 20, 25, 30]
REQUIRED_COLUMNS = ['Group', 'Replicate', 'Initial_fish', '0h', '24h', '48h',
                    '72h', '96h', '120h', 'Alive_120h']


def project_original_records(mother):
    """Validate and retain all 21 tanks from the original seven-dose experiment.

    The projection keeps the original ten-column input schema and row order.
    No original-batch dose is excluded from the primary analysis.
    """
    import re
    required = REQUIRED_COLUMNS + ['Salinity_ppt', 'Record_set']
    if missing := set(required) - set(mother.columns):
        raise ValueError(f'Missing original-batch mortality fields: {sorted(missing)}')
    raw = mother.copy()
    expected_groups = {f'S{dose}' for dose in [0, 5, 10, 15, 20, 25, 30]}
    if len(raw) != 21 or set(raw.Group) != expected_groups:
        raise ValueError('Shared mortality source requires exactly 21 original-batch tanks at 0/5/10/15/20/25/30 ppt; independent 18 ppt is excluded.')
    if not raw.Record_set.eq('Original batch').all():
        raise ValueError('Only the Original batch record set can supply LC50 observations.')
    for column in REQUIRED_COLUMNS[1:] + ['Salinity_ppt']:
        raw[column] = pd.to_numeric(raw[column], errors='raise')
        values = raw[column].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'Original-batch counts, doses and replicates must be finite integers: {column}')
    if any(not re.fullmatch(r'S(?:0|5|10|15|20|25|30)', str(group)) for group in raw.Group):
        raise ValueError('Original-batch group labels are not canonical salinity identifiers.')
    if not raw.Salinity_ppt.eq(raw.Group.str[1:].astype(int)).all():
        raise ValueError('Original-batch group labels and salinity values disagree.')
    if (raw.duplicated(['Group', 'Replicate']).any()
            or not raw.groupby('Group').Replicate.apply(lambda x: set(x) == {1, 2, 3}).all()):
        raise ValueError('Each original-batch salinity requires distinct replicate tanks 1, 2 and 3.')
    counts = raw[[f'{t}h' for t in TIMES]].to_numpy()
    if (not raw.Initial_fish.eq(50).all() or not raw['0h'].eq(0).all()
            or not np.all(np.diff(counts, axis=1) >= 0)
            or not np.all((counts >= 0) & (counts <= 50))
            or not (raw['120h'] + raw.Alive_120h).eq(raw.Initial_fish).all()):
        raise ValueError('Original-batch tank sizes or cumulative deaths do not reconcile.')
    selected = raw.loc[:, REQUIRED_COLUMNS].copy()
    validate_records(selected)
    return selected


def validate_records(raw, doses=DOSES):
    """Validate the source experiment without adding or removing observations."""
    if missing := set(REQUIRED_COLUMNS) - set(raw.columns):
        raise ValueError(f'Missing original mortality fields: {sorted(missing)}')
    raw = raw.copy()
    for column in REQUIRED_COLUMNS[1:]:
        raw[column] = pd.to_numeric(raw[column], errors='raise')
        if not np.isfinite(raw[column]).all() or not np.equal(raw[column], np.floor(raw[column])).all():
            raise ValueError(f'Original mortality counts/replicates must be finite integers: {column}')
    if list(doses) not in [DOSES, SIX_DOSE_SENSITIVITY]:
        raise ValueError('Only the complete original experiment or named six-dose sensitivity is supported.')
    if len(raw) != 3 * len(doses) or set(raw.Group) != {f'S{dose}' for dose in doses}:
        raise ValueError(f'LC50 requires exactly three original tanks at each requested dose: {doses}.')
    if (not raw.groupby('Group').size().eq(3).all()
            or raw.duplicated(['Group', 'Replicate']).any()
            or not raw.groupby('Group').Replicate.apply(lambda x: set(x) == {1, 2, 3}).all()):
        raise ValueError('The original salinity experiment requires distinct tanks 1, 2 and 3 per dose.')
    if not raw.Initial_fish.eq(50).all():
        raise ValueError('Every original tank must contain 50 fish initially.')
    counts = raw[[f'{t}h' for t in TIMES]].to_numpy()
    if not (np.all(np.diff(counts, axis=1) >= 0) and np.all((counts >= 0) & (counts <= 50))):
        raise ValueError('Cumulative deaths must be monotone and within the original tank size.')
    if not raw['0h'].eq(0).all() or not (raw['120h'] + raw.Alive_120h).eq(raw.Initial_fish).all():
        raise ValueError('Original baseline/final cumulative counts do not reconcile.')
    if not np.array_equal(raw['96h'], raw['120h']):
        raise ValueError('The original analysis requires unchanged cumulative counts from 96 to 120 h.')
    raw['Salinity_ppt'] = raw.Group.str[1:].astype(int)
    raw['Tank_ID'] = raw.Group + '_R' + raw.Replicate.astype(str)
    return raw


def prepare_timecourse(raw, doses=DOSES):
    """Expand each original tank to six times and compute tank/time summaries."""
    raw = validate_records(raw, doses)
    long = raw.melt(
        id_vars=['Group', 'Replicate', 'Tank_ID', 'Salinity_ppt', 'Initial_fish', 'Alive_120h'],
        value_vars=[f'{t}h' for t in TIMES], var_name='Time_h', value_name='Cumulative_deaths')
    long.Time_h = long.Time_h.str[:-1].astype(int)
    long = long.sort_values(['Salinity_ppt', 'Replicate', 'Time_h'])
    long['Surviving_fish'] = long.Initial_fish - long.Cumulative_deaths
    long['Survival_pct'] = 100 * long.Surviving_fish / long.Initial_fish
    long['Cumulative_mortality_pct'] = 100 - long.Survival_pct
    long['Interval_deaths'] = long.groupby('Tank_ID').Cumulative_deaths.diff().fillna(0).astype(int)
    summary = long.groupby(['Salinity_ppt', 'Time_h']).agg(
        Tanks=('Tank_ID', 'size'), Initial_fish=('Initial_fish', 'sum'),
        Cumulative_deaths=('Cumulative_deaths', 'sum'), Surviving_fish=('Surviving_fish', 'sum'),
        Survival_pct=('Survival_pct', 'mean'), Survival_min_pct=('Survival_pct', 'min'),
        Survival_max_pct=('Survival_pct', 'max'), Survival_SD_pct=('Survival_pct', 'std')).reset_index()
    summary['Cumulative_mortality_pct'] = 100 - summary.Survival_pct
    return long, summary, long.loc[long.Time_h.eq(96)].copy()


def fit(frame, link='logit'):
    linkobj = sm.families.links.Logit() if link == 'logit' else sm.families.links.Probit()
    X = sm.add_constant(frame.Salinity_ppt.to_numpy(dtype=float))
    y = np.column_stack([frame.Cumulative_deaths, frame.Initial_fish-frame.Cumulative_deaths])
    model = sm.GLM(y, X, family=sm.families.Binomial(link=linkobj)).fit(maxiter=200, tol=1e-11)
    if not model.converged or model.params[1] <= 0:
        raise ValueError('Mortality GLM did not converge to a positive salinity slope.')
    return model


def negative_log_likelihood(target, log_slope, x, deaths, totals):
    z = np.exp(log_slope) * (x-target)
    return float(np.sum(totals*np.logaddexp(0, z)-deaths*z))


def profile_loss(target, x, deaths, totals, minimum_nll):
    opt = minimize_scalar(lambda b: negative_log_likelihood(target, b, x, deaths, totals),
                          bounds=(-8, 4), method='bounded', options={'xatol': 1e-11})
    if not opt.success:
        raise ValueError('LC50 profile likelihood optimization did not converge.')
    return 2*(opt.fun-minimum_nll)-chi2.ppf(.95, 1)


def profile_likelihood(endpoint, model):
    """Conditional profile interval and 121-point profile, with original bounds."""
    lc50 = float(-model.params[0]/model.params[1])
    x = endpoint.Salinity_ppt.to_numpy()
    deaths = endpoint.Cumulative_deaths.to_numpy()
    totals = endpoint.Initial_fish.to_numpy()
    minimum = negative_log_likelihood(lc50, np.log(model.params[1]), x, deaths, totals)
    def profile(target):
        return profile_loss(target, x, deaths, totals, minimum)
    interval = [float(brentq(profile, 10, lc50)), float(brentq(profile, lc50, 25))]
    grid = np.linspace(interval[0]-.5, interval[1]+.5, 121)
    table = pd.DataFrame({'LC50_ppt': grid,
        'Twice_loglik_loss': [profile(z)+chi2.ppf(.95, 1) for z in grid]})
    return interval, table


def model_curves(logit, probit):
    grid = np.linspace(0, 30, 601)
    prediction = logit.get_prediction(sm.add_constant(grid)).summary_frame()
    return pd.DataFrame({'Salinity_ppt': grid, 'Mortality_pct': 100*prediction['mean'],
        'Conditional_lower95_pct': 100*prediction.mean_ci_lower,
        'Conditional_upper95_pct': 100*prediction.mean_ci_upper,
        'Probit_mortality_pct': 100*probit.predict(sm.add_constant(grid))})


def exact_tank_resampling(endpoint):
    """Enumerate an exact equally weighted within-dose empirical tank bootstrap.

    Three tanks are sampled with replacement at every dose. Identical-outcome
    strata are integrated out: their 27 possible ordered draws have the same
    likelihood contribution. Only S15/S20 vary in these data, so 27**2 = 729
    equally weighted reduced draws represent the full seven-dose bootstrap.
    Draw indices are distinct; fitted values need not be distinct.
    """
    groups = [part for _, part in endpoint.groupby('Salinity_ppt')]
    if any(len(part) != 3 for part in groups):
        raise ValueError('Exact tank resampling requires three tanks at every dose.')
    options = [list(product(range(3), repeat=3))
               if part[['Cumulative_deaths', 'Initial_fish']].drop_duplicates().shape[0] > 1
               else [(0, 1, 2)] for part in groups]
    boot = []
    for draws in product(*options):
        sample = pd.concat([part.iloc[list(indices)] for part, indices in zip(groups, draws)])
        model = fit(sample)
        boot.append(float(-model.params[0]/model.params[1]))
    return pd.DataFrame({'Enumeration': np.arange(1, len(boot)+1), 'LC50_ppt': boot})


def leave_one_tank_out(endpoint):
    records = []
    for idx, row in endpoint.iterrows():
        model = fit(endpoint.drop(index=idx))
        records.append({'Omitted_tank': row.Tank_ID, 'LC50_ppt': float(-model.params[0]/model.params[1])})
    return pd.DataFrame(records)


def model_diagnostics(endpoint, model):
    return pd.DataFrame({'Tank_ID': endpoint.Tank_ID, 'Salinity_ppt': endpoint.Salinity_ppt.to_numpy(),
        'Deaths': endpoint.Cumulative_deaths.to_numpy(), 'Initial_fish': endpoint.Initial_fish.to_numpy(),
        'Fitted_mortality_probability': model.fittedvalues, 'Pearson_residual': model.resid_pearson,
        'Deviance_residual': model.resid_deviance})


def analyze_records(raw, source_sha256, doses=DOSES):
    """Return all original tables and inference metadata without writing files."""
    long, summary, endpoint = prepare_timecourse(raw, doses)
    model, probit = fit(endpoint), fit(endpoint, 'probit')
    lc50 = float(-model.params[0]/model.params[1])
    lc50_probit = float(-probit.params[0]/probit.params[1])
    interval, profile = profile_likelihood(endpoint, model)
    resampling = exact_tank_resampling(endpoint)
    boot_ci = np.quantile(resampling.LC50_ppt.to_numpy(), [.025, .975]).tolist()
    loo = leave_one_tank_out(endpoint)
    info = {
        'status': 'author confirmed cumulative deaths in the same three replicate tanks per salinity throughout follow-up',
        'input_rows': len(raw), 'long_records': len(long), 'groups': list(doses), 'tanks_per_group': 3,
        'fish_per_tank': 50, 'total_initial_fish': int(raw.Initial_fish.sum()), 'endpoint_h': 96,
        'unchanged_96_to_120h': True, 'contains_18_ppt_observations': False,
        'logit_intercept': float(model.params[0]), 'logit_slope': float(model.params[1]),
        'LC50_ppt': lc50, 'conditional_profile95_ppt': interval, 'probit_LC50_ppt': lc50_probit,
        'tank_resampling_percentile95_sensitivity_ppt': boot_ci, 'resampling_fits': len(resampling),
        'resampling_note': 'Exact equally weighted reduced draws: 27 ordered draws for each of the two variable strata (15 and 20 ppt); all identical-outcome strata integrated out. Adding 5 ppt leaves 729 reduced draws.',
        'leave_one_tank_out_range_ppt': [float(loo.LC50_ppt.min()), float(loo.LC50_ppt.max())],
        'model_predicted_mortality_at18_pct': float(100*model.predict([[1, 18]])[0]),
        'pearson_chi2': float(model.pearson_chi2), 'residual_df': int(model.df_resid),
        'pearson_dispersion': float(model.pearson_chi2/model.df_resid), 'deviance': float(model.deviance),
        'all_rows_retained': True, 'all_fits_converged': True,
        'inference_note': 'Exploratory LC50. Likelihood CI and mean-response band assume independent binomial fish outcomes within tanks and a linear salinity logit, without a tank random effect. Three tanks per group and sparse intermediate salinities limit uncertainty estimation. Identical-outcome strata contribute no empirical resampling variance; this does not establish zero biological uncertainty. Tank-resampling percentiles are sensitivity bounds, not independent validation.',
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__},
        'source_sha256': source_sha256,
    }
    fits = pd.DataFrame([
        {'Model': 'Grouped-binomial logit', 'LC50_ppt': lc50, 'lower95': interval[0], 'upper95': interval[1], 'Interval_type': 'Conditional profile likelihood'},
        {'Model': 'Grouped-binomial probit (sensitivity)', 'LC50_ppt': lc50_probit, 'lower95': np.nan, 'upper95': np.nan, 'Interval_type': 'Not estimated'},
        {'Model': 'Within-salinity tank resampling (sensitivity)', 'LC50_ppt': lc50, 'lower95': boot_ci[0], 'upper95': boot_ci[1], 'Interval_type': 'Empirical percentiles; n=3 tanks/group'},
    ])
    tables = {
        f'pretest_tank_timecourse_{len(long)}_records.tsv': long,
        'salinity_survival_timecourse.tsv': summary,
        'salinity_original_lc50_curve.tsv': model_curves(model, probit),
        'lc50_profile_likelihood.tsv': profile,
        f'lc50_tank_resampling_{len(resampling)}_equally_weighted.tsv': resampling,
        'lc50_leave_one_tank_out.tsv': loo,
        'lc50_model_diagnostics.tsv': model_diagnostics(endpoint, model),
        'salinity_lc50_model_fit.tsv': fits,
    }
    return tables, info


def run(ctx, task='mortality'):
    source = ctx.require(INPUTS_BY_TASK[task])['salinity_original_seven_group_records']
    raw = project_original_records(read_table(source))
    tables, info = analyze_records(raw, sha256(source))
    info['all_source_rows_used_for_lc50'] = True
    info['source_projection'] = {
        'source_rows': 21, 'selected_rows': 21,
        'selection': 'All original-batch tanks at 0/5/10/15/20/25/30 ppt',
        'excluded_groups': [], 'excluded_rows': 0,
        'reason': 'The seven doses belong to the same original gradient experiment.',
        'selected_columns': REQUIRED_COLUMNS,
    }
    for name, table in tables.items():
        write_table(table, ctx.output(name))
    write_json(ctx.output('mortality_analysis.json'), info)
    return {'tables': list(tables), 'summary': 'mortality_analysis.json', 'LC50_ppt': info['LC50_ppt'],
            'input_rows': info['input_rows'], 'resampling_fits': info['resampling_fits']}
