import json

from celestial_detector.gmm_classifier import fit_gmm, load_gmm_boundaries, save_gmm_boundaries


def _samples():
    return [
        (25.0, 10.0),
        (26.0, 12.0),
        (30.0, 100.0),
        (29.0, 120.0),
        (15.0, 10_000.0),
        (14.0, 8_000.0),
        (0.0, 10_000.0),
        (1.0, 12_000.0),
    ]


def test_gmm_boundary_round_trip_preserves_classification(tmp_path):
    model = fit_gmm(_samples())
    boundary_path = tmp_path / 'gmm_boundaries.json'

    save_gmm_boundaries(model, boundary_path)
    loaded_model = load_gmm_boundaries(boundary_path)

    with boundary_path.open(encoding='utf-8') as stream:
        payload = json.load(stream)

    assert len(payload['components']) == 4
    assert len(payload['pairwise_boundaries']) == 6
    assert [
        loaded_model.classify(observations, error_km)
        for observations, error_km in _samples()
    ] == [
        model.classify(observations, error_km)
        for observations, error_km in _samples()
    ]


def test_live_observation_classification_returns_a_seeded_cluster():
    model = fit_gmm(_samples())

    cluster_id = model.classify_observations(25.0)

    assert cluster_id in {1, 2, 3, 4}