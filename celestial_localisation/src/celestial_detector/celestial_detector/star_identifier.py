"""Star pattern identification against a known catalogue.

This is a stub: it does not yet perform lost-in-space plate solving. Future
work should integrate a catalogue matcher (e.g. tetra3) here to resolve
each detected centroid into a known star identity (e.g. "HIP_32349").
Until then every star is reported as unidentified, which the localizer
should treat as unusable for absolute position solving.
"""


def identify_stars(star_detections):
    for star in star_detections:
        star['object_id'] = 'UNKNOWN'
    return star_detections
