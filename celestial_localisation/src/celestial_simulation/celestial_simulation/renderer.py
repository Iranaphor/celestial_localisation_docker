"""Stellarium rendering and equirectangular composition."""
import numpy as np

from celestial_simulation.projection import compose_equirectangular
from celestial_simulation.stellarium_browser import StellariumBrowserBridge


class StellariumRenderer:
    def __init__(
        self,
        engine_js,
        engine_wasm,
        data_root,
        browser_executable='',
        face_size=512,
        timeout_seconds=60.0,
        panorama_width=2048,
        panorama_height=1024,
        face_field_of_view=95.0,
        show_object_labels=False,
        enable_landscape=True,
        landscape_key='guereins',
    ):
        self.panorama_width = int(panorama_width)
        self.panorama_height = int(panorama_height)
        self.face_field_of_view = float(face_field_of_view)
        self._bridge = StellariumBrowserBridge(
            engine_js=engine_js,
            engine_wasm=engine_wasm,
            data_root=data_root,
            browser_executable=browser_executable,
            face_size=face_size,
            timeout_seconds=timeout_seconds,
            show_object_labels=show_object_labels,
            enable_landscape=enable_landscape,
            landscape_key=landscape_key,
        )

    def render(self, latitude, longitude, altitude, timestamp_ms, yaw_degrees=0.0):
        faces = self._bridge.render_faces(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            timestamp_ms=timestamp_ms,
            field_of_view_degrees=self.face_field_of_view,
            yaw_degrees=yaw_degrees,
        )
        image = compose_equirectangular(
            faces,
            self.panorama_width,
            self.panorama_height,
            self.face_field_of_view,
        )
        if image.shape != (self.panorama_height, self.panorama_width, 3):
            raise RuntimeError('Stellarium renderer returned an unexpected panorama shape')
        return np.ascontiguousarray(image, dtype=np.uint8)

    def close(self):
        self._bridge.close()
