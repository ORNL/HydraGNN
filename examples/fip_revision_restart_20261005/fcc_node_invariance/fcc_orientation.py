"""Raw crystal-to-sample FCC equivalence, with fixed fitting preprocessing."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
from scipy.spatial.transform import Rotation


def symmetries():
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            matrix[np.arange(3), permutation] = signs
            if np.linalg.det(matrix) > 0.5:
                result.append(matrix)
    matrices = np.stack(result)
    assert matrices.shape == (24, 3, 3)
    assert np.allclose(matrices @ matrices.transpose(0, 2, 1), np.eye(3))
    assert np.allclose(np.linalg.det(matrices), 1)
    return matrices


SYMMETRIES = symmetries()
IDENTITY_OPERATOR = int(np.flatnonzero(np.all(SYMMETRIES == np.eye(3), axis=(1, 2)))[0])
NORMALS = np.repeat(
    [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], 3, axis=0
) / np.sqrt(3)
DIRECTIONS = (
    np.array(
        [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
            [0, -1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, -1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [-1, 1, 0],
        ]
    )
    / np.sqrt(2)
)
assert np.allclose((NORMALS * DIRECTIONS).sum(axis=1), 0)


def matrices_from_eulers(eulers):
    # Same active Bunge product as the frozen dataset source, with radians.
    return Rotation.from_euler("ZXZ", np.asarray(eulers, dtype=np.float64)).as_matrix()


def sorted_schmid(rotations):
    normals = np.einsum("nij,kj->nki", rotations, NORMALS)
    directions = np.einsum("nij,kj->nki", rotations, DIRECTIONS)
    factors = np.abs(normals[:, :, 0] * directions[:, :, 0]) / (1 + 1e-12) ** 2
    return np.sort(factors, axis=1)[:, ::-1]


def equivalent_features(raw, operators):
    """R -> R S independently by grain. Select q sign nearest stored q."""
    original = raw.x.cpu().numpy()
    rotations = matrices_from_eulers(original[:, :3])
    changed = rotations @ SYMMETRIES[np.asarray(operators)]
    xyzw = Rotation.from_matrix(changed).as_quat()
    quaternion = xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    dot = (quaternion.astype(np.float64) * original[:, 3:7].astype(np.float64)).sum(
        axis=1
    )
    quaternion[dot < 0] *= -1
    dot = (quaternion.astype(np.float64) * original[:, 3:7].astype(np.float64)).sum(
        axis=1
    )
    assert np.all(dot >= 0)
    schmid = sorted_schmid(changed).astype(np.float32)
    expected = original[:, 28:40]
    error = float(
        np.max(np.abs(schmid.astype(np.float64) - expected.astype(np.float64)))
    )
    assert error < 2e-6, error
    assert np.isfinite(quaternion).all() and np.isfinite(schmid).all()
    assert np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1)) < 2e-7
    # Preserve source floating-point tensors exactly for the identity action,
    # after checking the independent physical reconstruction above.
    identity = np.asarray(operators) == IDENTITY_OPERATOR
    if np.any(identity):
        normalized_original = original[identity, 3:7].astype(np.float64)
        normalized_original /= np.linalg.norm(normalized_original, axis=1)[:, None]
        assert np.max(np.abs(quaternion[identity] - normalized_original)) < 1e-6
    quaternion[identity] = original[identity, 3:7]
    schmid[identity] = expected[identity]
    return quaternion, schmid, error


def source_quaternion_parity(raw):
    values = raw.x.cpu().numpy()
    xyzw = Rotation.from_matrix(matrices_from_eulers(values[:, :3])).as_quat()
    rebuilt = xyzw[:, [3, 0, 1, 2]]
    original = values[:, 3:7].astype(np.float64)
    original /= np.linalg.norm(original, axis=1)[:, None]
    rebuilt[np.sum(rebuilt * original, axis=1) < 0] *= -1
    error = float(np.max(np.abs(rebuilt - original)))
    assert error < 1e-6
    return error


def transformed_equivalent(ds, raw, operators):
    """Only selected orientation channels are rebuilt before fixed scaling."""
    import torch

    quaternion, schmid, error = equivalent_features(raw, operators)
    clone = raw.clone()
    clone.x = raw.x.clone()
    clone.x[:, 3:7] = torch.from_numpy(quaternion)
    clone.x[:, 28:40] = torch.from_numpy(schmid)
    out = ds._transform(clone)
    original = ds._transform(raw)
    assert torch.equal(out.x[:, 16:], original.x[:, 16:])
    for key in original.keys():
        if key == "x":
            continue
        a, b = original[key], out[key]
        if torch.is_tensor(a):
            assert torch.equal(a, b) or torch.allclose(
                a, b, rtol=0, atol=0, equal_nan=True
            ), key
        else:
            assert a == b, key
    return out, error


def fixtures(dataset_module):
    eulers = np.random.default_rng(107).uniform(0, 2 * np.pi, size=(32, 3))
    rotations = matrices_from_eulers(eulers)
    expected = dataset_module._schmid12_from_eulers(eulers)
    source_error = float(np.max(np.abs(sorted_schmid(rotations) - expected)))
    symmetry_error = max(
        float(np.max(np.abs(sorted_schmid(rotations @ s) - expected)))
        for s in SYMMETRIES
    )
    assert source_error < 1e-12 and symmetry_error < 1e-12
    return {
        "proper_cubic_rotations": 24,
        "fixture_grains": 32,
        "source_schmid_max_error": source_error,
        "all24_sorted_schmid_invariance_max_error": symmetry_error,
        "crystal_to_sample_action": "R -> R S, right multiplication",
        "quaternion_order": "wxyz; sign chosen nearest each grain original stored quaternion",
        "identity_policy": "Source tensors preserved exactly after physical reconstruction parity",
        "sample_rotation": False,
        "labels_topology_geometry_changed": False,
        "Euler_storage": "Unselected raw Euler columns kept as source metadata; selected quaternion and Schmid channels are rebuilt from physical R S",
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))
