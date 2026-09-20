import numpy as np

from celestial_detector.star_identifier import StarIdentifier


class MatchingSolver:
    database_properties = {'star_catalog': 'hip_main'}

    def solve_from_centroids(self, centroids, **kwargs):
        return {
            'Matches': len(centroids),
            'Prob': 0.0,
            'RMSE': 1.0,
            'matched_centroids': centroids,
            'matched_catID': np.arange(100, 100 + len(centroids)),
            'matched_stars': [[10.0, 20.0, 2.0]] * len(centroids),
        }


class RejectingSolver:
    database_properties = {'star_catalog': 'hip_main'}

    def solve_from_centroids(self, centroids, **kwargs):
        return None


def _nearby_detections():
    return [
        {
            'pixel_x': 100.0 + index * 8.0,
            'pixel_y': 400.0 + (index % 2) * 6.0,
            'brightness': 10.0 - index,
        }
        for index in range(4)
    ]


def test_identifies_catalogue_ids_from_equirectangular_centroids():
    identifier = StarIdentifier(solver=MatchingSolver())

    detections = identifier.identify(_nearby_detections(), (2048, 1024))

    assert {star['object_id'] for star in detections} == {
        'HIP_100', 'HIP_101', 'HIP_102', 'HIP_103',
    }
    assert all(star['match_count'] == 4 for star in detections)


def test_failed_solution_keeps_detections_unknown():
    identifier = StarIdentifier(solver=RejectingSolver())

    detections = identifier.identify(_nearby_detections(), (2048, 1024))

    assert all(star['object_id'] == 'UNKNOWN' for star in detections)