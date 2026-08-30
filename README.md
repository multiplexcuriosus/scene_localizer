# scene_localizer

ROS 2 nodes for table-scene localization, 2D-to-3D ball localization, and ball
trajectory estimation.

## Standalone RGB table-bump map

`tools/detect_table_bumps.py` detects dark connected components on a bright
table in one calibrated RGB image, intersects their camera rays with a known
bump-top height, and writes JSON, CSV, segmentation-overlay, brightness-
histogram, top-down-map, and metadata artifacts. It is a non-ROS tool and does
not change any runtime node or plotting behavior.

Install its ordinary Python dependencies (using a virtual environment is
recommended):

```bash
python3 -m pip install numpy opencv-python PyYAML matplotlib
```

Example:

```bash
python3 tools/detect_table_bumps.py \
  --image /path/to/table_rgb.png \
  --crop 600 230 1060 550 \
  --crop-upper-right 120 230 \
  --camera-info /path/to/camera_info.yaml \
  --table-pose /path/to/T_camera_table.yaml \
  --bump-height-m 0.025 \
  --table-size 0.60 1.20 \
  --segmentation-mode hard_threshold \
  --dark-l-threshold 110 \
  --output-dir /tmp/table_bump_map
```

The `600 230 1060 550` crop is only an initial estimate around the visible
bump field. Tune it and the segmentation thresholds from
`segmentation_overlay.png`; a component touching the crop boundary is rejected
by default. Crop and `--exclude-rect` coordinates are always full-image pixel
coordinates, and the image is never resized. Use `--rectified` only when the
input image has already had lens distortion removed.

`--crop-upper-right A B` removes a triangle from the already-rectangular crop.
In crop-local coordinates its vertices are `(W-A, 0)`, `(W, 0)`, and `(W, B)`,
where `W` is the crop width. Thus, `A` is the distance left along the top edge
and `B` is the distance down the right edge. The excluded triangle is black
with a magenta outline in `segmentation_overlay.png`.

Pixels are converted to 8-bit CIELAB and detection uses the Lab L brightness
channel. `--segmentation-mode hard_threshold` applies only `L <
--dark-l-threshold`, followed by morphological opening/closing and the spatial
gates. `otsu` chooses a hard threshold from the valid-region histogram.
`local_contrast` retains the Gaussian-background, bright-table, and local-dark
tests. `brightness_histogram.png` and the `brightness_histogram` section in
`metadata.json` record the 256-bin histogram, Otsu split, class statistics, and
separation metrics.

The table-pose file must describe `T_camera_table`, so table-frame points are
transformed into `camera_color_optical_frame`. The preferred explicit format
is:

```yaml
parent_frame: camera_color_optical_frame
child_frame: table_frame
translation: [tx, ty, tz]
quaternion_xyzw: [qx, qy, qz, qw]
```

A PoseStamped-style YAML with `header.frame_id`, `pose.position`, and
`pose.orientation` is also accepted. Incompatible frame metadata (for example,
a `T_base_cam` calibration) is rejected rather than reinterpreted. Camera YAML
may use ROS `camera_matrix`/`distortion_coefficients` mappings or simple `K`/`D`
arrays.

## Raw latency tracing

The 2D-to-3D and trajectory nodes can publish raw
`intercept_latency_monitor/msg/LatencyTrace` records. Tracing is disabled by
default and does not create a publisher or take timestamps on the data path
until enabled. Existing input/output topics, message types, QoS, node names,
parameters, and estimation algorithms are unchanged.

Both instrumented nodes support:

| Parameter | Default | Meaning |
|---|---|---|
| `enable_latency_trace` | `false` | Enable raw trace publication. |
| `latency_trace_topic` | stage-specific `/intercept_trace/...` topic | Output trace topic. |
| `latency_run_id` | empty | Run identifier copied into every trace. |
| `latency_modality` | `vision` | Modality copied into every trace. |

Localization traces use stage `localization_2d_to_3d` and include the incoming
detection source timestamp, callback receipt, localization start/end, result
validity, and (when published) the 3D output publication timestamp.

Trajectory traces use stage `trajectory_estimation` and include 3D observation
receipt, fit start/end, fit acceptance, source timestamp, and output publication
status. `scalar_value` contains finite fit RMS; `detail_json` contains the fit
reason, observation count, finite RMS, and output timestamp when available.
Invalid/rejected results are emitted as raw traces too. No percentiles, rolling
statistics, or per-sample INFO logs are produced by these nodes.

Example:

```bash
ros2 launch scene_localizer all_scene_localizer_nodes.launch.py \
  enable_latency_trace:=true latency_run_id:=trial_001 latency_modality:=vision
```

The standalone trajectory launch exposes the same four parameters, using
`latency_trace_topic` for its trace topic.
