import csv
import json

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


def test_filters_harmful_star_from_identification(tmp_path):
    metrics_path = tmp_path / 'simulated_location_filter_samples*.csv'
    fieldnames = ('record_type', 'error_distance_meters', 'identified_star_ids', 'cluster_id')
    rows = []
    for repetition in range(3):
        rows.append({
            'record_type': 'step',
            'error_distance_meters': 1_000_000,
            'identified_star_ids': json.dumps(['HIP_100']),
            'cluster_id': 3,
        })
        rows.append({
            'record_type': 'step',
            'error_distance_meters': 1_000,
            'identified_star_ids': json.dumps([]),
            'cluster_id': 3,
        })
    for repetition in range(3):
        rows.append({
            'record_type': 'step',
            'error_distance_meters': 1_000,
            'identified_star_ids': json.dumps(['HIP_100']),
            'cluster_id': 1,
        })
        rows.append({
            'record_type': 'step',
            'error_distance_meters': 100_000,
            'identified_star_ids': json.dumps([]),
            'cluster_id': 1,
        })
    for file_index in range(2):
        source_path = tmp_path / f'simulated_location_filter_samples_{file_index + 1}.csv'
        with source_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[file_index::2])
    unrelated_path = tmp_path / 'other_samples.csv'
    unrelated_path.write_text('not,a,metrics,file\n', encoding='utf-8')

    identifier = StarIdentifier(
        solver=MatchingSolver(),
        star_benefit_metrics_path=metrics_path,
    )

    detections = identifier.identify(_nearby_detections(), (2048, 1024))

    assert identifier.excluded_star_ids == {'HIP_100'}
    assert [path.name for path in identifier.star_benefit_source_paths] == [
        'simulated_location_filter_samples_1.csv',
        'simulated_location_filter_samples_2.csv',
    ]
    assert detections[0]['object_id'] == 'UNKNOWN'
    assert detections[0]['catalogue_id'] == 'HIP_100'
    assert detections[0]['excluded_by_star_benefit'] is True
    assert detections[1]['object_id'] == 'HIP_101'


def test_medium_error_group_can_be_enabled(tmp_path):
    metrics_path = tmp_path / 'random_localisation_metrics.csv'
    fieldnames = ('record_type', 'error_distance_meters', 'identified_star_ids', 'cluster_id')
    rows = [
        {
            'record_type': 'step',
            'error_distance_meters': 1_000_000,
            'identified_star_ids': json.dumps(['HIP_100']),
            'cluster_id': 2,
        }
        for _ in range(3)
    ]
    rows.extend({
        'record_type': 'step',
        'error_distance_meters': 1_000,
        'identified_star_ids': json.dumps([]),
        'cluster_id': 2,
    } for _ in range(3))
    with metrics_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    identifier = StarIdentifier(
        solver=MatchingSolver(),
        star_benefit_metrics_path=metrics_path,
        star_benefit_use_high_error_group=False,
        star_benefit_use_medium_error_group=True,
    )

    detections = identifier.identify(_nearby_detections(), (2048, 1024))

    assert identifier.excluded_star_ids == {'HIP_100'}
    assert detections[0]['object_id'] == 'UNKNOWN'