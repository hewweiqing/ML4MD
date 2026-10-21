"""Small offline tests; no GPU, model training, downloads or PDF generation."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze as pca


class MethodTests(unittest.TestCase):
    def test_cosine_definition(self):
        h = np.array([[1., 0.], [1., 1.]])
        actual = pca.relative(h, np.eye(2))
        np.testing.assert_allclose(actual, [[1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2)]])

    def test_relative_ignores_positive_row_scaling(self):
        rng = np.random.default_rng(3)
        h, anchors = rng.normal(size=(12, 5)), rng.normal(size=(4, 5))
        np.testing.assert_allclose(pca.relative(h, anchors),
            pca.relative(h * np.arange(1, 13)[:, None], anchors * 20), atol=1e-14)

    def test_undefined_cosine_is_not_silently_replaced(self):
        for h, anchors in ((np.zeros((3, 4)), np.ones((2, 4))),
                           (np.ones((3, 4)), np.zeros((2, 4))),
                           (np.full((3, 4), np.nan), np.ones((2, 4)))):
            with self.assertRaises(ValueError):
                pca.relative(h, anchors)

    def test_pool_then_relative_is_not_relative_then_pool(self):
        atoms = np.array([[1., 0.], [0., 2.]])
        anchors = np.eye(2)
        self.assertFalse(np.allclose(pca.relative(atoms.mean(0, keepdims=True), anchors),
                                     pca.relative(atoms, anchors).mean(0)))

    def test_validation_cannot_change_fitted_PCA(self):
        rng = np.random.default_rng(4)
        train, val = rng.normal(size=(40, 7)), rng.normal(size=(10, 7))
        first, _ = pca.fit_pca(train, dict(val=val))
        second, _ = pca.fit_pca(train, dict(val=val * 1000 + 999))
        for field in ('mean', 'components', 'explained_variance_ratio'):
            np.testing.assert_array_equal(first[field], second[field])

    def test_rank_one_does_not_invent_extra_coordinates(self):
        h = np.arange(20)[:, None] * np.array([[1., 2., 3., 4.]])
        fit, coordinates = pca.fit_pca(h, dict(val=h))
        self.assertEqual(fit['meaningful_PC_count'], 1)
        np.testing.assert_array_equal(coordinates['val'][:, 1:], np.zeros((20, 2)))

    def test_axis_percentages_are_reference_variance(self):
        h = np.random.default_rng(8).normal(size=(50, 6))
        fit, _ = pca.fit_pca(h, dict(val=h[:5]))
        total = h.var(axis=0, ddof=1).sum()
        np.testing.assert_allclose(fit['explained_variance_ratio'], fit['explained_variance'] / total)

    def test_zero_reference_variance_is_rejected(self):
        with self.assertRaises(ValueError):
            pca.fit_pca(np.ones((5, 4)), dict(val=np.ones((2, 4))))

    def test_anchors_are_deterministic_distinct_and_element_stratified(self):
        numbers = np.repeat([1, 8, 26], 100)
        a = pca.atomic_anchors(numbers)
        np.testing.assert_array_equal(a, pca.atomic_anchors(numbers))
        self.assertEqual(len(set(a)), 100)
        np.testing.assert_array_equal(numbers[a[:3]], [1, 8, 26])

    def test_checksum_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input.npz').write_bytes(b'original')
            manifest = dict(sha256={'input.npz': pca.sha256(root / 'input.npz')})
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            pca.verify_inputs(root)
            (root / 'input.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                pca.verify_inputs(root)

    def test_nonempty_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'keep.txt').write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                pca.run(ROOT / 'data', root, ['crabnet'], plot=False)
            self.assertEqual((root / 'keep.txt').read_text(encoding='utf-8'), 'keep')


class PublishedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / 'data'
        cls.manifest = pca.verify_inputs(cls.data)

    def test_same_real_cohort_and_no_test_partition(self):
        previous = None
        for model in pca.MODELS:
            identity = pca.load(self.data / model / 'identities.npz')
            self.assertEqual(len(identity['train_material_ids']), 2014)
            self.assertEqual(len(identity['val_material_ids']), 252)
            self.assertEqual(int(identity['val_labels'].sum()), 126)
            current = [identity[k] for k in ('train_material_ids', 'val_material_ids', 'val_labels')]
            if previous is not None:
                for a, b in zip(previous, current):
                    np.testing.assert_array_equal(a, b)
            previous = current
            self.assertFalse(any('test' in key for key in identity))

    def test_exact_published_anchor_reference_and_display_identities(self):
        for model in pca.MODELS:
            identity = pca.load(self.data / model / 'identities.npz')
            for level in ('individual', 'material'):
                labels = pca.selection(identity, level)
                self.assertEqual(len(labels), len(identity[f'{level}_display_ids']))
                self.assertEqual(len(identity[f'{level}_anchor_ids']), 100)

    def test_DIRECT_is_default_and_authors_source_is_pinned(self):
        self.assertEqual(self.manifest['default_anchor_method'], 'direct')
        self.assertEqual(pca.sha256(ROOT / 'support/third_party/anchor_selection.py'),
                         self.manifest['direct']['source_sha256'])
        import inspect
        self.assertEqual(inspect.signature(pca.run).parameters['anchors'].default, 'direct')

    def test_DIRECT_training_identities_counts_and_original_rows(self):
        for model in pca.MODELS:
            identity = pca.load(self.data / model / 'identities.npz')
            keys = self.manifest['models'][model]['states']
            selected, audit = pca.direct_selection(self.data, model, identity, keys)
            for level in ('individual', 'material'):
                self.assertEqual(audit[level]['reference_checkpoint'], 'ControlS20')
                self.assertEqual(audit[level]['requested_count'], 100)
                self.assertEqual(audit[level]['actual_count'], 100)
                self.assertFalse(set(selected[f'{level}_ids']) & set(identity[f'val_{level}_ids']))
                refs = {v: i for i, v in enumerate(identity[f'{level}_reference_ids'])}
                matched = [(i, refs[v]) for i, v in enumerate(selected[f'{level}_ids']) if v in refs]
                for k, key in enumerate(keys):
                    original = pca.load(self.data / model / f'{key}.npz')[f'{level}_train']
                    for a, r in matched:
                        np.testing.assert_array_equal(selected[f'{level}_vectors'][k, a], original[r])

    def test_DIRECT_does_not_change_reference_or_display_identities(self):
        for model in pca.MODELS:
            identity = pca.load(self.data / model / 'identities.npz')
            before = {k: v.copy() for k, v in identity.items()}
            pca.direct_selection(self.data, model, identity, self.manifest['models'][model]['states'])
            for key in before:
                np.testing.assert_array_equal(before[key], identity[key])

    def test_DIRECT_state_order_mismatch_fails(self):
        identity = pca.load(self.data / 'crabnet/identities.npz')
        with self.assertRaises(AssertionError):
            pca.direct_selection(self.data, 'crabnet', identity,
                                 self.manifest['models']['crabnet']['states'][::-1])

    def test_DIRECT_reselection_rejects_degenerate_population_before_importing_MAML(self):
        for features in (np.ones((101, 4)), np.full((101, 4), np.nan)):
            with self.assertRaises(ValueError):
                pca.reselect_direct(features, self.data)

    def test_reference_PDF_and_saved_results_checksums(self):
        reference = ROOT / 'support/reference'
        for name, digest in pca.read(reference / 'checksums.json').items():
            self.assertEqual(pca.sha256(reference / name), digest)

    def test_DIRECT_reference_coordinates_reconstruct_without_changing_PCA_population(self):
        reference = ROOT / 'support/reference/direct'
        for name, digest in pca.read(reference / 'checksums.json').items():
            self.assertEqual(pca.sha256(reference / name), digest)
        with threadpool_limits(limits=2):
            for model in pca.MODELS:
                selected = pca.load(self.data / 'direct' / model / 'anchors.npz')
                for level in ('individual', 'material'):
                    saved = pca.load(reference / f'{model}_{level}_relative.npz')
                    refs = []
                    for i, key in enumerate(self.manifest['models'][model]['states']):
                        state = pca.load(self.data / model / f'{key}.npz')
                        anchors = selected[f'{level}_vectors'][i]
                        val = pca.relative(state[f'{level}_val'], anchors)
                        coordinates = (val - saved['mean']) @ saved['components'].T
                        np.testing.assert_allclose(coordinates, saved['coordinates'][i], rtol=1e-10, atol=1e-10)
                        refs.append(pca.relative(state[f'{level}_train'], anchors))
                    np.testing.assert_allclose(np.concatenate(refs).mean(0), saved['mean'], rtol=1e-10, atol=1e-10)

    def test_original_coordinates_reconstructed_from_released_features(self):
        # Verify every state without refitting expensive full PCA in the unit tests.
        with threadpool_limits(limits=2):
            for model in pca.MODELS:
                metadata = self.manifest['models'][model]
                for level in ('individual', 'material'):
                    for view in ('raw', 'relative'):
                        old = pca.load(ROOT / 'support/reference' / f'{model}_{level}_{view}.npz')
                        np.testing.assert_array_equal(old['checkpoint_keys'], metadata['states'])
                        refs = []
                        for i, key in enumerate(metadata['states']):
                            state = pca.load(self.data / model / f'{key}.npz')
                            train, val = state[f'{level}_train'].astype(float), state[f'{level}_val'].astype(float)
                            if view == 'relative':
                                train = pca.relative(train, state[f'{level}_anchors'])
                                val = pca.relative(val, state[f'{level}_anchors'])
                            coords = (val - old['mean']) @ old['components'].T
                            np.testing.assert_allclose(coords, old['coordinates'][i], rtol=1e-10, atol=1e-10)
                            refs.append(train)
                        np.testing.assert_allclose(np.concatenate(refs).mean(0), old['mean'], rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
