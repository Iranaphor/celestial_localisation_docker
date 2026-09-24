"""Offline star identification for equirectangular sky maps.

The plate solver works on perspective images, while the detector publishes
centroids from a full-sky equirectangular image.  This module bridges those
representations by projecting detections into overlapping virtual perspective
tiles before passing them to tetra3.
"""
import math

import numpy as np

from celestial_detector.star_benefit_filter import StarBenefitFilter


_UNKNOWN_ID = 'UNKNOWN'
_DEFAULT_CATALOGUE = 'hip_main'


class StarIdentifier:
    """Identify detected stars with an offline tetra3 database."""

    def __init__(
        self,
        database_path='',
        fov_degrees=30.0,
        fov_max_error_degrees=3.0,
        tile_size=512,
        match_radius=0.02,
        match_threshold=0.001,
        min_matches=4,
        pattern_checking_stars=8,
        solver=None,
        star_benefit_metrics_path='',
        star_benefit_min_score=-0.1,
        star_benefit_min_present_count=3,
        star_benefit_min_absent_count=2,
        star_benefit_use_high_error_group=True,
        star_benefit_use_medium_error_group=False,
    ):
        self.fov_degrees = float(fov_degrees)
        self.fov_max_error_degrees = float(fov_max_error_degrees)
        self.tile_size = int(tile_size)
        self.match_radius = float(match_radius)
        self.match_threshold = float(match_threshold)
        self.min_matches = int(min_matches)
        self.pattern_checking_stars = int(pattern_checking_stars)
        self._solver = solver
        self.error = ''
        self.star_benefit_filter = StarBenefitFilter.empty()
        self.star_benefit_error = ''
        if star_benefit_metrics_path:
            try:
                self.star_benefit_filter = StarBenefitFilter.from_csv(
                    star_benefit_metrics_path,
                    min_benefit_score=star_benefit_min_score,
                    min_present_count=int(star_benefit_min_present_count),
                    min_absent_count=int(star_benefit_min_absent_count),
                    use_high_error_group=bool(star_benefit_use_high_error_group),
                    use_medium_error_group=bool(star_benefit_use_medium_error_group),
                )
            except (OSError, TypeError, ValueError) as error:
                self.star_benefit_error = f'could not load star benefit metrics: {error}'

        if self._solver is not None:
            return

        try:
            import tetra3
        except ImportError as error:
            self.error = f'tetra3 is unavailable: {error}'
            return

        try:
            load_database = database_path or 'default_database'
            self._solver = tetra3.Tetra3(load_database=load_database)
        except (OSError, RuntimeError, ValueError) as error:
            self.error = f'could not load star database: {error}'

    @property
    def enabled(self):
        return self._solver is not None

    @property
    def catalogue_name(self):
        if self._solver is None:
            return _DEFAULT_CATALOGUE
        properties = getattr(self._solver, 'database_properties', {})
        if callable(properties):
            properties = properties()
        return properties.get('star_catalog', _DEFAULT_CATALOGUE)

    @property
    def excluded_star_ids(self):
        return self.star_benefit_filter.excluded_star_ids

    @property
    def star_benefit_source_paths(self):
        return self.star_benefit_filter.source_paths

    def identify(self, star_detections, panorama_size):
        """Annotate detections with catalogue IDs where a solution is valid."""
        for star in star_detections:
            star['object_id'] = _UNKNOWN_ID

        if not self.enabled or len(star_detections) < self.min_matches:
            return star_detections

        panorama_width, panorama_height = panorama_size
        if panorama_width <= 0 or panorama_height <= 0:
            return star_detections

        directions = np.asarray([
            self._direction_from_pixel(
                star['pixel_x'], star['pixel_y'], panorama_width, panorama_height
            )
            for star in star_detections
        ])
        best_matches = {}

        for centre_index in range(len(star_detections)):
            tile_indices, tile_centroids = self._make_tile(
                directions, star_detections, centre_index
            )
            if len(tile_indices) < self.min_matches:
                continue

            brightness_order = sorted(
                range(len(tile_indices)),
                key=lambda index: star_detections[tile_indices[index]].get('brightness', 0.0),
                reverse=True,
            )
            ordered_indices = [tile_indices[index] for index in brightness_order]
            ordered_centroids = tile_centroids[brightness_order]
            result = self._solve(ordered_centroids)
            if result is None:
                continue

            match_count = int(result.get('Matches') or 0)
            if match_count < self.min_matches:
                continue
            false_probability = result.get('Prob')
            if false_probability is not None and float(false_probability) > self.match_threshold:
                continue

            matched_centroids = result.get('matched_centroids')
            matched_catalogue_ids = result.get('matched_catID')
            matched_stars = result.get('matched_stars')
            if matched_centroids is None or matched_catalogue_ids is None:
                continue

            false_probability = float(false_probability) if false_probability is not None else 1.0
            rmse = result.get('RMSE')
            rmse = float(rmse) if rmse is not None else float('inf')
            quality = (match_count, -false_probability, -rmse)
            used_indices = set()

            for match_index, matched_centroid in enumerate(matched_centroids):
                if match_index >= len(matched_catalogue_ids):
                    break
                candidate_position = np.asarray(matched_centroid, dtype=np.float64)
                distances = np.linalg.norm(ordered_centroids - candidate_position, axis=1)
                for used_position in used_indices:
                    distances[used_position] = float('inf')
                source_position = int(np.argmin(distances))
                if not np.isfinite(distances[source_position]):
                    continue
                if distances[source_position] > max(2.0, self.tile_size * self.match_radius):
                    continue
                used_indices.add(source_position)

                source_index = ordered_indices[source_position]
                previous = best_matches.get(source_index)
                if previous is not None and previous[0] >= quality:
                    continue

                catalogue_id = self._format_catalogue_id(matched_catalogue_ids[match_index])
                metadata = {
                    'object_id': catalogue_id,
                    'catalogue_id': catalogue_id,
                    'match_count': match_count,
                    'match_probability': false_probability,
                    'match_rmse_arcseconds': rmse,
                }
                if matched_stars is not None and match_index < len(matched_stars):
                    matched_star = matched_stars[match_index]
                    if len(matched_star) >= 3:
                        metadata['catalogue_ra'] = float(matched_star[0])
                        metadata['catalogue_dec'] = float(matched_star[1])
                        metadata['catalogue_magnitude'] = float(matched_star[2])
                best_matches[source_index] = (quality, metadata)

        for source_index, (_, metadata) in best_matches.items():
            star_detections[source_index].update(metadata)
            if self.star_benefit_filter.excludes(metadata['object_id']):
                star_detections[source_index]['object_id'] = _UNKNOWN_ID
                star_detections[source_index]['excluded_by_star_benefit'] = True
        return star_detections

    def _solve(self, centroids):
        try:
            return self._solver.solve_from_centroids(
                np.asarray(centroids, dtype=np.float64),
                size=(self.tile_size, self.tile_size),
                fov_estimate=self.fov_degrees,
                fov_max_error=self.fov_max_error_degrees,
                pattern_checking_stars=min(self.pattern_checking_stars, len(centroids)),
                match_radius=self.match_radius,
                match_threshold=self.match_threshold,
                return_matches=True,
            )
        except (RuntimeError, ValueError, TypeError):
            return None

    def _make_tile(self, directions, star_detections, centre_index):
        forward = directions[centre_index]
        reference_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, reference_up))) > 0.95:
            reference_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(reference_up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        tangent = math.tan(math.radians(self.fov_degrees) / 2.0)

        depth = directions @ forward
        horizontal = directions @ right
        vertical = directions @ up
        visible = (
            (depth > 0.0)
            & (np.abs(horizontal) <= depth * tangent)
            & (np.abs(vertical) <= depth * tangent)
        )
        tile_indices = np.flatnonzero(visible).tolist()
        tile_centroids = np.column_stack((
            (1.0 - (vertical[visible] / depth[visible] / tangent)) * 0.5 * (self.tile_size - 1),
            ((horizontal[visible] / depth[visible] / tangent) + 1.0) * 0.5 * (self.tile_size - 1),
        ))
        return tile_indices, tile_centroids

    @staticmethod
    def _direction_from_pixel(pixel_x, pixel_y, width, height):
        azimuth = (float(pixel_x) / float(width)) * 2.0 * math.pi
        elevation = math.pi / 2.0 - (float(pixel_y) / float(height)) * math.pi
        cos_elevation = math.cos(elevation)
        return np.array([
            math.cos(azimuth) * cos_elevation,
            math.sin(azimuth) * cos_elevation,
            math.sin(elevation),
        ], dtype=np.float64)

    def _format_catalogue_id(self, catalogue_id):
        catalogue_name = self.catalogue_name.lower()
        if catalogue_name == 'hip_main':
            return f'HIP_{int(catalogue_id)}'
        if catalogue_name == 'bsc5':
            return f'BSC_{int(catalogue_id)}'
        if catalogue_name == 'tyc_main':
            values = np.asarray(catalogue_id).reshape(-1).tolist()
            return 'TYC_' + '_'.join(str(int(value)) for value in values)
        return str(catalogue_id)


def identify_stars(star_detections, panorama_size=None, identifier=None):
    """Backward-compatible entry point used by the detector node."""
    if identifier is None:
        identifier = StarIdentifier()
    if panorama_size is None:
        for star in star_detections:
            star['object_id'] = _UNKNOWN_ID
        return star_detections
    return identifier.identify(star_detections, panorama_size)
