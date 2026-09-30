import json

from celestial_detector.gmm_classifier import (
    cluster_filename,
    fit_gmm,
    load_gmm_boundaries,
    save_gmm_boundaries,
)


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

    assert [
        (seed['observations'], seed['error_km'])
        for seed in payload['seed_locations']
    ] == [(10.0, 8.0), (25.0, 200.0), (8.0, 5000.0)]
    assert len(payload['components']) == 3
    assert len(payload['pairwise_boundaries']) == 3
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

    assert cluster_id in {1, 2, 3}


def test_cluster_filename_includes_the_cluster_name():
    assert cluster_filename(1) == 'cluster1_lowerror.png'
    assert cluster_filename(3) == 'cluster3_higherror.png'