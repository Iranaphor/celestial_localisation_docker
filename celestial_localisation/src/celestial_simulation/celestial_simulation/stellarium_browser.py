"""Headless browser bridge for Stellarium Web Engine."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from celestial_simulation.projection import cube_faces


class RendererError(RuntimeError):
    """Raised when the Stellarium browser renderer cannot produce an image."""


class _AssetRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        request_path = unquote(urlparse(self.path).path)
        server = self.server
        if request_path in ('/', '/index.html'):
            self._send_bytes(server.html, 'text/html; charset=utf-8')
            return
        if request_path == '/engine.js':
            self._send_file(server.engine_js, 'application/javascript')
            return
        if request_path == '/engine.wasm':
            self._send_file(server.engine_wasm, 'application/wasm')
            return
        if request_path.startswith('/data/'):
            relative = Path(request_path[len('/data/'):])
            if relative.is_absolute() or '..' in relative.parts:
                self.send_error(404)
                return
            candidate = (server.data_root / relative).resolve()
            try:
                candidate.relative_to(server.data_root)
            except ValueError:
                self.send_error(404)
                return
            if candidate.is_file():
                self._send_file(candidate, 'application/octet-stream')
                return
        self.send_error(404)

    def _send_file(self, path, content_type):
        try:
            data = Path(path).read_bytes()
        except OSError:
            self.send_error(404)
            return
        self._send_bytes(data, content_type)

    def _send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format_string, *args):
        return


class _AssetServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StellariumBrowserBridge:
    def __init__(
        self,
        engine_js,
        engine_wasm,
        data_root,
        browser_executable='',
        face_size=512,
        timeout_seconds=60.0,
        show_object_labels=False,
        enable_landscape=True,
        landscape_key='guereins',
    ):
        self.engine_js = Path(engine_js).expanduser() if engine_js else None
        self.engine_wasm = Path(engine_wasm).expanduser() if engine_wasm else None
        self.data_root = Path(data_root).expanduser() if data_root else None
        self.browser_executable = browser_executable
        self.face_size = int(face_size)
        self.timeout_seconds = float(timeout_seconds)
        self.show_object_labels = bool(show_object_labels)
        self.enable_landscape = bool(enable_landscape)
        self.landscape_key = str(landscape_key)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._server = None

    def render_faces(self, latitude, longitude, altitude, timestamp_ms, field_of_view_degrees):
        self._ensure_started()
        self._page.evaluate(
            'args => window.__configureObserver(args)',
            {
                'latitude': float(latitude),
                'longitude': float(longitude),
                'altitude': float(altitude),
                'timestampMs': float(timestamp_ms),
                'fieldOfViewDegrees': float(field_of_view_degrees),
            },
        )
        self._page.wait_for_timeout(25)

        faces = {}
        for face in cube_faces():
            azimuth = math.degrees(math.atan2(face.center[1], face.center[0])) % 360.0
            elevation = math.degrees(math.asin(face.center[2]))
            suppress_downward_sky = self.enable_landscape and face.name == 'nadir'
            if suppress_downward_sky:
                self._page.evaluate('() => window.__setDownwardSkyVisible(false)')
                self._page.wait_for_timeout(350)
            try:
                self._page.evaluate(
                    'args => window.__renderFace(args.azimuth, args.elevation)',
                    {'azimuth': azimuth, 'elevation': elevation},
                )
                self._page.wait_for_timeout(100)
                png = self._page.locator('#stel-canvas').screenshot(type='png')
                image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RendererError(f'failed to decode the {face.name} browser screenshot')
                if image.shape[:2] != (self.face_size, self.face_size):
                    image = cv2.resize(image, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)
                faces[face.name] = image
            finally:
                if suppress_downward_sky:
                    self._page.evaluate('() => window.__setDownwardSkyVisible(true)')
                    self._page.wait_for_timeout(350)
        return faces

    def _ensure_started(self):
        if self._page is not None:
            return
        self._validate_assets()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RendererError(
                'Playwright is not installed; install the browser bridge runtime'
            ) from error

        self._server = self._create_server()
        try:
            self._playwright = sync_playwright().start()
            launch_options = {
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--enable-webgl',
                    '--ignore-gpu-blocklist',
                    '--use-angle=swiftshader',
                ],
            }
            if self.browser_executable:
                launch_options['executable_path'] = self.browser_executable
            self._browser = self._playwright.chromium.launch(**launch_options)
            self._context = self._browser.new_context(
                viewport={'width': self.face_size, 'height': self.face_size},
                device_scale_factor=1,
            )
            self._page = self._context.new_page()
            self._page.goto(
                self._server.url,
                wait_until='load',
                timeout=int(self.timeout_seconds * 1000),
            )
            self._page.wait_for_function(
                'window.__stelState !== "loading"',
                timeout=int(self.timeout_seconds * 1000),
            )
            state = self._page.evaluate('window.__stelState')
            if state != 'ready':
                error = self._page.evaluate('window.__stelError || "unknown engine error"')
                raise RendererError(f'Stellarium Web Engine failed to initialize: {error}')
            self._page.wait_for_function(
                'window.__dataReady === true || window.__stelState === "error"',
                timeout=int(self.timeout_seconds * 1000),
            )
            state = self._page.evaluate('window.__stelState')
            if state != 'ready':
                error = self._page.evaluate('window.__stelError || "unknown data error"')
                raise RendererError(f'Stellarium data failed to load: {error}')
        except Exception:
            self.close()
            raise

    def _validate_assets(self):
        if self.engine_js is None or not self.engine_js.is_file():
            raise RendererError(
                f'Stellarium engine JavaScript asset is missing: {self.engine_js or "<unset>"}'
            )
        if self.engine_wasm is None or not self.engine_wasm.is_file():
            raise RendererError(
                f'Stellarium engine WebAssembly asset is missing: {self.engine_wasm or "<unset>"}'
            )
        if self.data_root is None or not self.data_root.is_dir():
            raise RendererError(
                f'Stellarium data root is missing: {self.data_root or "<unset>"}'
            )
        if not self._data_sources():
            raise RendererError(f'no supported Stellarium data sources found below {self.data_root}')
        if self.face_size < 32:
            raise RendererError('renderer face size must be at least 32 pixels')

    def _data_sources(self):
        candidates = (
            ('stars', 'stars', ''),
            ('skycultures', 'skycultures/western', 'western'),
            ('dsos', 'dso', ''),
            ('landscapes', f'landscapes/{self.landscape_key}', self.landscape_key),
            ('milkyway', 'surveys/milkyway', ''),
            ('planets', 'surveys/sso/moon', 'moon'),
            ('planets', 'surveys/sso/sun', 'sun'),
        )
        return [
            {'module': module, 'path': path, 'key': key}
            for module, path, key in candidates
            if (self.data_root / path).exists()
            and (self.enable_landscape or module != 'landscapes')
        ] if self.data_root else []

    def _create_server(self):
        html = self._build_html(
            self._data_sources(),
            self.show_object_labels,
            self.enable_landscape,
            self.landscape_key,
        )
        server = _AssetServer(('127.0.0.1', 0), _AssetRequestHandler)
        server.html = html.encode('utf-8')
        server.engine_js = self.engine_js
        server.engine_wasm = self.engine_wasm
        server.data_root = self.data_root.resolve()
        server.url = f'http://127.0.0.1:{server.server_address[1]}/index.html'
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def _build_html(self, data_sources, show_object_labels, enable_landscape, landscape_key):
        html = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #000; } #stel-canvas { display: block; width: 100%; height: 100%; }</style>
</head>
<body>
<canvas id="stel-canvas"></canvas>
<script>
const originalGetContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function(type, attributes) {
    if (type === 'webgl' || type === 'experimental-webgl' || type === 'webgl2') {
        attributes = Object.assign({}, attributes || {}, {preserveDrawingBuffer: true});
    }
    return originalGetContext.call(this, type, attributes);
};
</script>
<script src="/engine.js"></script>
<script>
window.__stelState = 'loading';
window.__stelError = null;
window.__dataReady = false;
window.__dataStableFrames = 0;
const dataSources = __DATA_SOURCES__;
const showObjectLabels = __SHOW_OBJECT_LABELS__;
const enableLandscape = __ENABLE_LANDSCAPE__;
const landscapeKey = __LANDSCAPE_KEY__;
function fail(error) {
  window.__stelState = 'error';
  window.__stelError = String(error && error.stack ? error.stack : error);
}
function checkDataReady() {
  if (window.__stelState !== 'ready') {
    window.requestAnimationFrame(checkDataReady);
    return;
  }
  const bars = window.__stel.core.progressbars || [];
  const failed = bars.find(bar => bar.error);
  if (failed) {
    fail(failed.error_msg || 'data source reported an error');
    return;
  }
  const pending = bars.some(bar => Number(bar.total) > 0 && Number(bar.value) < Number(bar.total));
  window.__dataStableFrames = pending ? 0 : window.__dataStableFrames + 1;
  if (window.__dataStableFrames >= 10) {
    window.__dataReady = true;
    return;
  }
  window.requestAnimationFrame(checkDataReady);
}
window.__configureObserver = function(args) {
  const stel = window.__stel;
  const observer = stel.observer;
  observer.latitude = args.latitude * stel.D2R;
  observer.longitude = args.longitude * stel.D2R;
  observer.elevation = args.altitude;
  observer.utc = stel.date2MJD(new Date(args.timestampMs));
  observer.roll = 0;
  stel.core.projection = 1;
  stel.core.fov = args.fieldOfViewDegrees * stel.D2R;
  stel.core.flip_view_vertical = false;
  stel.core.flip_view_horizontal = false;
};
window.__renderFace = function(azimuth, elevation) {
  const stel = window.__stel;
  const azimuthRadians = azimuth * stel.D2R;
  const elevationRadians = elevation * stel.D2R;
  stel.lookAt([
    Math.cos(azimuthRadians) * Math.cos(elevationRadians),
    Math.sin(azimuthRadians) * Math.cos(elevationRadians),
    Math.sin(elevationRadians)
  ], 0);
  stel.observer.roll = 0;
  stel._core_update();
  stel._core_render(window.innerWidth, window.innerHeight, 1);
};
window.__setDownwardSkyVisible = function(visible) {
    const core = window.__stel.core;
    [core.stars, core.milkyway, core.dsos, core.planets,
     core.comets, core.minor_planets, core.satellites].forEach(module => {
        if (module) module.visible = visible;
    });
};
try {
  StelWebEngine({
    wasmFile: '/engine.wasm',
    canvas: document.getElementById('stel-canvas'),
    onReady: function(stel) {
      try {
        window.__stel = stel;
        const core = stel.core;
        core.projection = 1;
                if (core.landscapes) {
                    core.landscapes.visible = enableLandscape;
                    core.landscapes.fog_visible = enableLandscape;
                }
        core.atmosphere.visible = false;
                core.cardinals.visible = false;
                core.stars.hints_visible = showObjectLabels;
                core.planets.hints_visible = showObjectLabels;
                core.dsos.hints_visible = showObjectLabels;
                core.comets.hints_visible = showObjectLabels;
                core.minor_planets.hints_visible = showObjectLabels;
                core.satellites.hints_visible = showObjectLabels;
                core.constellations.labels_visible = showObjectLabels;
        dataSources.forEach(source => {
          const module = core[source.module];
          if (module) {
            module.addDataSource({url: '/data/' + source.path, key: source.key || 0});
          }
        });
                if (enableLandscape && core.landscapes) {
                    core.landscapes.current_id = landscapeKey;
                }
        window.__stelState = 'ready';
        window.requestAnimationFrame(checkDataReady);
      } catch (error) {
        fail(error);
      }
    }
  });
} catch (error) {
  fail(error);
}
</script>
</body>
</html>
'''
        return (
            html
            .replace('__DATA_SOURCES__', json.dumps(data_sources, separators=(',', ':')))
            .replace('__SHOW_OBJECT_LABELS__', json.dumps(show_object_labels))
            .replace('__ENABLE_LANDSCAPE__', json.dumps(enable_landscape))
            .replace('__LANDSCAPE_KEY__', json.dumps(landscape_key))
        )

    def close(self):
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None

    def __del__(self):
        self.close()
