"""Ephemeris predictions for known solar-system bodies via Astropy.

Star ephemerides are not yet implemented here because celestial_detector
does not currently resolve star identities to catalogue entries (see
celestial_detector/star_identifier.py). Only SUN and MOON are predictable
today; extend `predict` once a star catalogue matcher is available.
"""
from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time
import astropy.units as u


class EphemerisProvider:
    """Predicts expected az/el for a celestial body given observer state and time."""

    SUPPORTED_BODIES = ('sun', 'moon')

    def predict(self, object_id, timestamp, latitude, longitude, altitude=0.0):
        if object_id.lower() not in self.SUPPORTED_BODIES:
            return None

        location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
        time = Time(timestamp, format='unix')
        frame = AltAz(obstime=time, location=location)
        body = get_body(object_id.lower(), time, location)
        altaz = body.transform_to(frame)
        return float(altaz.az.deg), float(altaz.alt.deg)
