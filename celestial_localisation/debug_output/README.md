# Debug output

This tracked file ensures the directory exists before Docker Compose starts.
Otherwise, Docker may create the bind-mount source as `root`, preventing the
non-root `ros` user from writing diagnostic images and JSON files.