#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
set -u

stellarium_source_dir="${CELESTIAL_STELLARIUM_SOURCE_DIR:-/opt/stellarium-web-engine}"
stellarium_output_dir="${CELESTIAL_STELLARIUM_RUNTIME_DIR:-/opt/stellarium}"
stellarium_build_dir="/tmp/stellarium-web-engine-build"

build_stellarium_assets() {
	if [ -f "${stellarium_output_dir}/stellarium-web-engine.js" ] \
		&& [ -f "${stellarium_output_dir}/stellarium-web-engine.wasm" ] \
		&& [ -d "${stellarium_output_dir}/data/stars" ] \
		&& [ -d "${stellarium_output_dir}/data/skycultures/western" ] \
		&& [ -d "${stellarium_output_dir}/data/dso" ] \
		&& [ -d "${stellarium_output_dir}/data/surveys/milkyway" ] \
		&& [ -d "${stellarium_output_dir}/data/surveys/sso/sun" ] \
		&& [ -d "${stellarium_output_dir}/data/surveys/sso/moon" ]; then
		return
	fi

	if [ ! -f "${stellarium_source_dir}/Makefile" ] \
		|| [ ! -d "${stellarium_source_dir}/apps/test-skydata" ]; then
		echo "Stellarium Web Engine source is missing: ${stellarium_source_dir}" >&2
		exit 1
	fi

	if ! command -v emscons >/dev/null 2>&1; then
		echo "emscons is required to build Stellarium Web Engine" >&2
		exit 1
	fi

	if [ -z "${EMSCRIPTEN_TOOL_PATH:-}" ] && command -v em-config >/dev/null 2>&1; then
		export EMSCRIPTEN_TOOL_PATH="$(em-config EMSCRIPTEN_ROOT)/tools"
	fi

	rm -rf "${stellarium_build_dir}"
	mkdir -p "${stellarium_build_dir}" "${stellarium_output_dir}"
	cp -a "${stellarium_source_dir}/." "${stellarium_build_dir}/"

	cd "${stellarium_build_dir}"
	emscons scons -j8 mode=release werror=0

	install -m 0644 build/stellarium-web-engine.js \
		"${stellarium_output_dir}/stellarium-web-engine.js"
	install -m 0644 build/stellarium-web-engine.wasm \
		"${stellarium_output_dir}/stellarium-web-engine.wasm"
	rm -rf "${stellarium_output_dir}/data"
	mkdir -p "${stellarium_output_dir}/data"
	cp -a apps/test-skydata/. "${stellarium_output_dir}/data/"
}

build_stellarium_assets

cd /home/ros/ros2_ws
colcon build --symlink-install
set +u
source install/setup.bash
set -u

exec ros2 launch celestial_bringup celestial_localisation.launch.py
