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

## Native event-camera ball localization

`event_ball_pipeline.launch.py` replaces only the RGB 2D-to-3D front end. It
starts one calibration adapter, one `ball_3d_pose_estimator`, and one unchanged
`ball_trajectory_estimator`:

```text
/openmv_cam/event_tracker/ball_2d_px
  -> /scene_localizer/event/ball_3d_table
  -> /scene/ball_trajectory_table
```

The event pixels are the raw native GENX320 coordinates: 320x320, top-left
origin, x right, y down. They are not rotated, mirrored, resized, cropped, or
otherwise reinterpreted. `event_ball_pipeline.yaml` sets
`input_pixels_are_rectified: false`, so the estimator applies
`cv2.undistortPoints()` exactly once using the adapter's raw K and D. The RGB
default remains `input_pixels_are_rectified: true`, preserving its legacy
direct-pinhole behavior for an already-rectified RGB detector/CameraInfo pair.
Do not set the parameter to false for pixels that were already rectified.

The adapter loads the solved event calibration YAML directly. It consumes the
serializer's top-level `camera_matrix`, `distortion_coefficients`,
`distortion_model`, `image_width`, `image_height`, and `T_table_camera` blocks.
The convention is `T_parent_child`: `T_table_camera` maps event-camera
coordinates into the calibrated table coordinates. The published
`PoseStamped` is computed as `T_camera_table = inverse(T_table_camera)`, has
`header.frame_id=event_camera` by default, and describes the table in that
camera frame. The adapter publishes:

- `/event_camera/camera_info`
- `/scene_localizer/event_camera/table_pose_camera`

The 2D-to-3D estimator publishes:

- `/scene_localizer/event/ball_3d_camera` in `event_camera`
- `/scene_localizer/event/ball_3d_table` in `table_frame`

The table output is a ball-center `geometry_msgs/PointStamped` on
`z_table=ball_radius`, retaining the incoming event detection timestamp. The
trajectory fit, middle-line intersection, `/scene/ball_trajectory_table`, and
the downstream GOTO_S/TRACK_S interfaces are unchanged.

Launch the event branch with the solved YAML:

```bash
ros2 launch scene_localizer event_ball_pipeline.launch.py \
  calibration_file:="$HOME/.ros/event_camera_calibration/genx320_calibration.yaml"
```

If `all_scene_localizer_nodes.launch.py` is used to supply the table-to-base
pose, TCP TF, and robot-base conversion, inhibit its RGB ball nodes so there is
only one trajectory publisher:

```bash
ros2 launch scene_localizer all_scene_localizer_nodes.launch.py \
  inhibit_ball_3d_pose_estimator:=true \
  inhibit_ball_trajectory_estimator:=true
```

The adapter defaults to requiring 320x320 calibration dimensions and rejects a
mismatch rather than scaling coordinates. Set `expected_image_width` or
`expected_image_height` to a non-positive value only to disable that explicit
check for a deliberately different native sensor.

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
