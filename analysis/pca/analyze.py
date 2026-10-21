"""Reproduce checkpoint-feature PCA; no model imports, training or test data.

Run: python analyze.py
Tests: python -m unittest discover -s support -p test_pca.py -v
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import importlib.util
import sys

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
MODELS = ('orb', 'alignn', 'crabnet')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def atomic_anchors(numbers, count=100, seed=8100):
    """Seeded element-stratified round robin, matching the published pilot."""
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(np.flatnonzero(numbers == z))) for z in np.unique(numbers)]
    selected = []
    while pools and len(selected) < count:
        for pool in pools:
            if pool and len(selected) < count:
                selected.append(pool.pop())
        pools = [pool for pool in pools if pool]
    return np.asarray(selected, dtype=int)


def stable_subset(parents, cap, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for parent in np.unique(parents):
        indices = np.flatnonzero(parents == parent)
        selected.extend(np.sort(rng.choice(indices, min(cap, len(indices)), replace=False)))
    return np.asarray(selected, dtype=int)


def families(numbers):
    groups = {
        'H': {1},
        'Alkali / alkaline earth': {3, 11, 19, 37, 55, 87, 4, 12, 20, 38, 56, 88},
        'Transition metals': set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113)),
        'Lanthanides / actinides': set(range(57, 72)) | set(range(89, 104)),
        'Chalcogens': {8, 16, 34, 52, 84, 116},
        'Halogens': {9, 17, 35, 53, 85, 117},
        'Noble gases': {2, 10, 18, 36, 54, 86, 118},
    }
    return np.array([next((name for name, elements in groups.items() if int(z) in elements),
                         'Other main group') for z in numbers])


def selection(identity, level):
    """Recalculate and verify all anchor/reference/display identities."""
    if level == 'individual':
        tr = identity['train_individual_ids']
        va = identity['val_individual_ids']
        anchors = atomic_anchors(identity['train_atomic_numbers'])
        refs = stable_subset(identity['train_individual_parents'], 1, 8101)
        display = stable_subset(identity['val_individual_parents'], 8, 8102)
        labels = families(identity['val_atomic_numbers'][display])
    elif level == 'material':
        tr = identity['train_material_ids']
        va = identity['val_material_ids']
        anchors = np.random.default_rng(8100).choice(len(tr), 100, replace=False)
        refs = np.arange(len(tr))
        display = np.arange(len(va))
        labels = np.where(identity['val_labels'], 'Metal', 'Non-metal')
    else:
        raise ValueError(f'Unknown level: {level}')
    if len(set(tr.tolist())) != len(tr) or len(set(va.tolist())) != len(va):
        raise ValueError('Repeated observation identities')
    if set(tr.tolist()) & set(va.tolist()):
        raise ValueError('Training/validation overlap')
    for role, ids in (('anchor', tr[anchors]), ('reference', tr[refs]), ('display', va[display])):
        np.testing.assert_array_equal(ids, identity[f'{level}_{role}_ids'])
    return labels


def direct_selection(data, model, identity, keys, verify=False):
    """Load frozen author-DIRECT identities, never substitute baseline anchors."""
    folder = data / 'direct' / model
    selected = load(folder / 'anchors.npz')
    population = load(folder / 'control_population.npz')
    audit = read(folder / 'selection.json')
    np.testing.assert_array_equal(selected['checkpoint_keys'], keys)
    for level in ('individual', 'material'):
        ids, indices = selected[f'{level}_ids'], selected[f'{level}_indices']
        train_ids = identity[f'train_{level}_ids']
        np.testing.assert_array_equal(population[f'{level}_ids'], train_ids)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(indices) < 3 or len(set(indices.tolist())) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(train_ids))):
            raise ValueError('Invalid or repeated DIRECT training indices')
        np.testing.assert_array_equal(ids, train_ids[indices])
        np.testing.assert_array_equal(ids, audit[level]['selected_ids'])
        if len(ids) != audit[level]['actual_count']:
            raise ValueError('DIRECT count does not match selection audit')
        vectors = selected[f'{level}_vectors']
        if vectors.shape != (len(keys), len(ids), population[f'{level}_features'].shape[1]):
            raise ValueError('DIRECT anchor shape/state mismatch')
        np.testing.assert_array_equal(vectors[keys.index('ControlS20')],
                                      population[f'{level}_features'][indices])
        if verify:
            actual = reselect_direct(population[f'{level}_features'], data)
            np.testing.assert_array_equal(actual, indices,
                err_msg='DIRECT selection drift: do not silently change the frozen anchor identities')
    return selected, audit


def reselect_direct(features, data):
    """Optional audit using the unchanged, pinned authors' wrapper and MAML."""
    specification = read(data / 'manifest.json')['direct']
    features = np.asarray(features)
    if (features.ndim != 2 or not np.isfinite(features).all()
            or len(np.unique(features, axis=0)) < specification['requested_count']):
        raise ValueError('DIRECT needs a finite population with enough distinct feature rows')
    source = ROOT / 'support/third_party/anchor_selection.py'
    if sha256(source) != specification['source_sha256']:
        raise ValueError('Pinned authors DIRECT wrapper checksum mismatch')
    module_spec = importlib.util.spec_from_file_location('_platonicrep_anchors', source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    if importlib.metadata.version('maml') != '2025.4.3':
        raise ValueError('DIRECT audit requires maml==2025.4.3 from support/requirements-direct.txt')
    rng_state = np.random.get_state()
    try:
        np.random.seed(specification['seed'])
        result = module.DIRECTSelector(n_anchors=specification['requested_count'],
            threshold_init=specification['threshold_init'],
            k_per_cluster=specification['k_per_cluster']).fit_transform(np.asarray(features, dtype=float))
    finally:
        np.random.set_state(rng_state)
    indices = np.asarray(result.anchor_indices, dtype=int)
    if len(set(indices.tolist())) != len(indices):
        raise ValueError('DIRECT returned repeated observation identities')
    np.testing.assert_array_equal(result.anchor_embeddings, features[indices])
    return indices


def relative(features, anchors, epsilon=1e-12):
    """Cosine to current-state anchors; never replace an undefined anchor."""
    features, anchors = np.asarray(features, dtype=float), np.asarray(anchors, dtype=float)
    norms = np.linalg.norm(features, axis=1)
    anchor_norms = np.linalg.norm(anchors, axis=1)
    if (not np.isfinite(features).all() or not np.isfinite(anchors).all()
            or np.any(norms <= epsilon) or np.any(anchor_norms <= epsilon)):
        raise ValueError('Undefined cosine: nonfinite or zero/near-zero feature vector')
    return (features / norms[:, None]) @ (anchors / anchor_norms[:, None]).T


def fit_pca(reference, validation):
    """Fit ONLY the concatenated training reference; project validation afterward."""
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or min(reference.shape) < 3 or not np.isfinite(reference).all():
        raise ValueError('PCA requires a finite reference with at least three rows/channels')
    if not np.any(reference != reference[0]):
        raise ValueError('All reference observations identical: PCA variance is undefined')
    fitted = PCA(n_components=3, svd_solver='full', whiten=False).fit(reference)
    tolerance = np.finfo(float).eps * max(reference.shape) * fitted.singular_values_[0]
    rank = int(np.count_nonzero(fitted.singular_values_ > tolerance))
    components = fitted.components_.copy()
    components[rank:] = 0.
    coordinates = {}
    for key, values in validation.items():
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != reference.shape[1] or not np.isfinite(values).all():
            raise ValueError(f'Invalid validation features: {key}')
        coordinates[key] = (values - fitted.mean_) @ components.T
    return dict(mean=fitted.mean_, components=components,
                explained_variance_ratio=np.nan_to_num(fitted.explained_variance_ratio_, nan=0.),
                explained_variance=fitted.explained_variance_, meaningful_PC_count=rank), coordinates


def verify_inputs(data):
    manifest = read(data / 'manifest.json')
    for relative_path, expected in manifest['sha256'].items():
        path = (data / relative_path).resolve()
        if data.resolve() not in path.parents or not path.is_file():
            raise ValueError(f'Missing or invalid data path: {relative_path}')
        if sha256(path) != expected:
            raise ValueError(f'Input checksum mismatch: {relative_path}')
    return manifest


def compare_reference(output, reference, models, anchors='baseline'):
    """Numerical regression, allowing the arbitrary sign of each PCA axis."""
    comparisons = []
    for model in models:
        for level in ('individual', 'material'):
            for view in ('raw', 'relative'):
                new = load(output / model / f'{level}_{view}.npz')
                folder = reference / 'direct' if anchors == 'direct' and view == 'relative' else reference
                old = load(folder / f'{model}_{level}_{view}.npz')
                for field in ('checkpoint_keys', 'ids', 'reference_ids', 'anchor_ids'):
                    np.testing.assert_array_equal(new[field], old[field])
                signs = np.where(np.sum(new['components'] * old['components'], axis=1) < 0, -1, 1)
                for field, value in (('mean', new['mean']),
                                     ('components', new['components'] * signs[:, None]),
                                     ('explained_variance_ratio', new['explained_variance_ratio']),
                                     ('coordinates', new['coordinates'] * signs)):
                    np.testing.assert_allclose(value, old[field], rtol=1e-8, atol=1e-8,
                                               err_msg=f'Reference mismatch: {model}/{level}/{view}/{field}')
                comparisons.append(dict(model=model, level=level, view=view,
                    maximum_coordinate_difference=float(np.abs(new['coordinates'] * signs - old['coordinates']).max()),
                    maximum_ratio_difference=float(np.abs(new['explained_variance_ratio'] - old['explained_variance_ratio']).max())))
    return comparisons


def compute(model, data, output, manifest, anchors='direct', verify_direct=False):
    metadata = manifest['models'][model]
    keys = metadata['states']
    if len(set(keys)) != len(keys) or keys.count('I') != 1 or keys.count('ControlS20') != 1:
        raise ValueError('Initialization and Control must each appear exactly once')
    identity = load(data / model / 'identities.npz')
    levels = {level: selection(identity, level) for level in ('individual', 'material')}
    direct, audit = direct_selection(data, model, identity, keys, verify_direct) if anchors == 'direct' else (None, None)
    collected = {level: {view: {'train': [], 'val': {}} for view in ('raw', 'relative')}
                 for level in levels}
    for index, key in enumerate(keys):
        state = load(data / model / f'{key}.npz')
        if str(state['checkpoint']) != key:
            raise ValueError(f'Wrong checkpoint inside {key}.npz')
        for level in levels:
            train, val, bank = (state[f'{level}_{role}'].astype(float) for role in ('train', 'val', 'anchors'))
            expected = len(identity[f'{level}_reference_ids']), len(identity[f'{level}_display_ids']), 100
            for matrix, count in zip((train, val, bank), expected):
                if matrix.shape != (count, metadata['dimension']) or not np.isfinite(matrix).all():
                    raise ValueError(f'Wrong feature shape/values: {model}/{key}/{level}')
            collected[level]['raw']['train'].append(train)
            collected[level]['raw']['val'][key] = val
            if direct is not None:
                bank = direct[f'{level}_vectors'][index]
            collected[level]['relative']['train'].append(relative(train, bank))
            collected[level]['relative']['val'][key] = relative(val, bank)
    folder = output / model
    folder.mkdir()
    summaries = []
    for level in levels:
        for view, values in collected[level].items():
            reference = np.concatenate(values['train'])
            fit, coordinates = fit_pca(reference, values['val'])
            np.savez_compressed(folder / f'{level}_{view}.npz', **fit,
                                coordinates=np.stack([coordinates[key] for key in keys]),
                                checkpoint_keys=np.array(keys), labels=levels[level],
                                ids=identity[f'{level}_display_ids'],
                                reference_ids=identity[f'{level}_reference_ids'],
                                anchor_ids=(direct[f'{level}_ids'] if direct is not None and view == 'relative'
                                            else identity[f'{level}_anchor_ids']))
            ratios = fit['explained_variance_ratio']
            summaries.append(dict(model=metadata['name'], level=level, view=view,
                                  reference_observations=len(reference), reference_states=len(keys),
                                  PC1=ratios[0], PC2=ratios[1], PC3=ratios[2],
                                  PC1_PC2=float(ratios[:2].sum()), PC1_PC2_PC3=float(ratios.sum())))
    print(f'{metadata["name"]}: four training-only PCA fits completed', flush=True)
    if audit is not None:
        (folder / 'anchor_selection.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return summaries


def run(data, output, models, plot=True, reference=None, anchors='direct', verify_direct=False):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is nonempty; choose a new --output: {output}')
    manifest = verify_inputs(data)
    if anchors not in ('direct', 'baseline') or (verify_direct and anchors != 'direct'):
        raise ValueError('Choose direct or baseline; DIRECT reselection requires direct mode')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(models={name: manifest['models'][name] for name in models}, seed=0,
                  input_manifest_sha256=sha256(data / 'manifest.json'),
                  input_checksums=manifest['sha256'], python=platform.python_version(),
                  versions={name: importlib.metadata.version(name) for name in (
                      'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'threadpoolctl')},
                  training=False, test_access=False, DIRECT=anchors == 'direct', anchor_method=anchors,
                  direct_selection_recomputed=verify_direct,
                  direct_provenance=manifest.get('direct') if anchors == 'direct' else None)
    if verify_direct:
        config['versions']['maml'] = importlib.metadata.version('maml')
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    with threadpool_limits(limits=2):
        for model in models:
            rows.extend(compute(model, data, output, manifest, anchors, verify_direct))
    with (output / 'PCA_explained_variance.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if reference is not None:
        comparisons = compare_reference(output, Path(reference), models, anchors)
        (output / 'reference_comparison.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
        print('Saved reference fits and coordinates agree within numerical tolerance', flush=True)
    if plot:
        from plot import plot_run
        plot_run(output, output / 'figures')
    (output / 'COMPLETE.json').write_text(json.dumps(dict(models=models, PCA_fits=len(rows),
        plotted=plot, native_training=False, original_data_untouched=True), indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, help='New directory (default: results/pca_direct or pca_baseline)')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--check-reference', action='store_true', help='Compare refitted PCA against bundled original results')
    parser.add_argument('--anchors', choices=('direct', 'baseline'), default='direct')
    parser.add_argument('--verify-direct-selection', action='store_true',
                        help='Re-run pinned author selection; requires support/requirements-direct.txt')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('Each model may be requested only once')
    run(args.data, args.output or ROOT / f'results/pca_{args.anchors}', args.models, plot=not args.no_plots,
        reference=ROOT / 'support/reference' if args.check_reference else None,
        anchors=args.anchors, verify_direct=args.verify_direct_selection)
