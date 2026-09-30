import numpy as np

from celestial_localiser.ephemeris import EphemerisProvider


def _provider_with_test_stars():
    provider = EphemerisProvider.__new__(EphemerisProvider)
    provider.star_coordinates = {
        'hip_1': (10.0, 20.0),
        'hip_2': (130.0, -30.0),
    }
    provider._star_coord_cache = {}
    provider._star_batch_cache = {}
    provider._time_cache = {}
    provider.star_catalogue = 'hip_main'
    provider.error = ''
    return provider


def test_batched_star_predictions_match_scalar_predictions():
    provider = _provider_with_test_stars()
    object_ids = ['HIP_1', 'HIP_2', 'HIP_1']
    timestamp = 1_700_000_000.1234567
    scalar = [
        provider.predict(object_id, timestamp, 23.0, -70.0, 0.0)
        for object_id in object_ids
    ]
    batched = provider.predict_many(object_ids, timestamp, 23.0, -70.0, 0.0)

    np.testing.assert_allclose(batched, scalar, rtol=0.0, atol=1e-10)
    assert len(provider._time_cache) == 1
    assert len(provider._star_coord_cache) == 2
    assert len(provider._star_batch_cache) == 1
