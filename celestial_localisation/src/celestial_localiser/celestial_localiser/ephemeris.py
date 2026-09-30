"""Ephemeris predictions for solar-system bodies and catalogue stars."""
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body
from astropy.time import Time
import astropy.units as u
import numpy as np


class EphemerisProvider:
    """Predicts expected az/el for a celestial body given observer state and time."""

    SUPPORTED_BODIES = ('sun', 'moon')

    def __init__(self, star_database_path=''):
        self.star_coordinates = {}
        self._star_coord_cache = {}
        self._star_batch_cache = {}
        self._time_cache = {}
        self.star_catalogue = ''
        self.error = ''
        self._load_star_catalogue(star_database_path)

    def predict(self, object_id, timestamp, latitude, longitude, altitude=0.0):
        location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
        time = self._time(timestamp)
        frame = AltAz(obstime=time, location=location)

        normalised_id = object_id.strip().lower()
        if normalised_id in self.SUPPORTED_BODIES:
            celestial_object = get_body(normalised_id, time, location)
        else:
            celestial_object = self._star_coord(normalised_id)
            if celestial_object is None:
                return None

        altaz = celestial_object.transform_to(frame)
        return float(altaz.az.deg), float(altaz.alt.deg)

    def predict_many(self, object_ids, timestamp, latitude, longitude, altitude=0.0):
        """Predict a fixed object list in one frame transform.

        The observer frame remains candidate-location dependent. Catalogue
        coordinates and the observation time are cached because they are
        constant throughout a solve.
        """
        object_ids = list(object_ids)
        results = [None] * len(object_ids)
        if not object_ids:
            return results

        location = EarthLocation(
            lat=latitude * u.deg,
            lon=longitude * u.deg,
            height=altitude * u.m,
        )
        time = self._time(timestamp)
        frame = AltAz(obstime=time, location=location)
        star_indices = []
        star_keys = []
        body_indices = []
        for index, object_id in enumerate(object_ids):
            normalised_id = object_id.strip().lower()
            if normalised_id in self.SUPPORTED_BODIES:
                body_indices.append((index, normalised_id))
                continue
            if self._star_coord(normalised_id) is not None:
                star_indices.append(index)
                star_keys.append(normalised_id)

        if star_indices:
            cache_key = tuple(star_keys)
            star_coordinates = self._star_batch_cache.get(cache_key)
            if star_coordinates is None:
                star_coordinates = SkyCoord(
                    ra=[self.star_coordinates[key][0] for key in star_keys] * u.deg,
                    dec=[self.star_coordinates[key][1] for key in star_keys] * u.deg,
                    frame='icrs',
                )
                self._star_batch_cache[cache_key] = star_coordinates
            altaz = star_coordinates.transform_to(frame)
            for index, azimuth, elevation in zip(
                star_indices,
                np.asarray(altaz.az.deg).reshape(-1),
                np.asarray(altaz.alt.deg).reshape(-1),
            ):
                results[index] = (float(azimuth), float(elevation))

        for index, body_name in body_indices:
            celestial_object = get_body(body_name, time, location)
            altaz = celestial_object.transform_to(frame)
            results[index] = (float(altaz.az.deg), float(altaz.alt.deg))
        return results

    def _time(self, timestamp):
        timestamp = float(timestamp)
        if timestamp not in self._time_cache:
            self._time_cache[timestamp] = Time(timestamp, format='unix')
        return self._time_cache[timestamp]

    def _star_coord(self, normalised_id):
        if normalised_id not in self.star_coordinates:
            return None
        if normalised_id not in self._star_coord_cache:
            right_ascension, declination = self.star_coordinates[normalised_id]
            self._star_coord_cache[normalised_id] = SkyCoord(
                ra=right_ascension * u.deg,
                dec=declination * u.deg,
                frame='icrs',
            )
        return self._star_coord_cache[normalised_id]

    def _load_star_catalogue(self, database_path):
        try:
            import tetra3
        except ImportError as error:
            self.error = f'tetra3 is unavailable: {error}'
            return

        try:
            load_database = database_path or 'default_database'
            solver = tetra3.Tetra3(load_database=load_database)
            properties = solver.database_properties
            self.star_catalogue = str(properties.get('star_catalog', '')).lower()
            catalogue_ids = solver.star_catalog_IDs
            star_table = solver.star_table
            if catalogue_ids is None:
                self.error = 'star database does not contain catalogue identifiers'
                return

            for star_index, catalogue_id in enumerate(catalogue_ids):
                star_key = self._catalogue_key(catalogue_id)
                if star_key is None:
                    continue
                self.star_coordinates[star_key] = (
                    float(star_table[star_index][0]) * 180.0 / 3.141592653589793,
                    float(star_table[star_index][1]) * 180.0 / 3.141592653589793,
                )
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            self.error = f'could not load star database: {error}'

    def _catalogue_key(self, catalogue_id):
        if self.star_catalogue == 'hip_main':
            return f'hip_{int(catalogue_id)}'
        if self.star_catalogue == 'bsc5':
            return f'bsc_{int(catalogue_id)}'
        if self.star_catalogue == 'tyc_main':
            values = [int(value) for value in catalogue_id]
            return 'tyc_' + '_'.join(str(value) for value in values)
        return None
