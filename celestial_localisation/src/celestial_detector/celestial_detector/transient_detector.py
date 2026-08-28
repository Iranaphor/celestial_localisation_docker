"""Aircraft/satellite rejection stub.

Real rejection requires temporal tracking (do candidate point sources move
with the star field, or independently?). This first-stage implementation
does not track objects across frames yet, so no point source is ever
classified as AIRCRAFT or SATELLITE. Wire this up once cross-frame tracking
(e.g. cv2.calcOpticalFlowPyrLK) is available.
"""


def classify_point_sources(star_detections):
    return star_detections
