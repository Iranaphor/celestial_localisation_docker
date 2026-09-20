"""Ephemeris predictions for solar-system bodies and catalogue stars."""
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body
from astropy.time import Time
import astropy.units as u


class EphemerisProvider:
    """Predicts expected az/el for a celestial body given observer state and time."""

    SUPPORTED_BODIES = ('sun', 'moon')

    def __init__(self, star_database_path=''):
        self.star_coordinates = {}
        self.star_catalogue = ''
        self.error = ''
        self._load_star_catalogue(star_database_path)

    def predict(self, object_id, timestamp, latitude, longitude, altitude=0.0):
        location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
        time = Time(timestamp, format='unix')
        frame = AltAz(obstime=time, location=location)

        normalised_id = object_id.strip().lower()
        if normalised_id in self.SUPPORTED_BODIES:
            celestial_object = get_body(normalised_id, time, location)
        else:
            coordinates = self.star_coordinates.get(normalised_id)
            if coordinates is None:
                return None
            right_ascension, declination = coordinates
            celestial_object = SkyCoord(
                ra=right_ascension * u.deg,
                dec=declination * u.deg,
                frame='icrs',
            )

        altaz = celestial_object.transform_to(frame)
        return float(altaz.az.deg), float(altaz.alt.deg)

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
