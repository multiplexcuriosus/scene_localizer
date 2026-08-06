# scene_localizer

ROS 2 nodes for table-scene localization, 2D-to-3D ball localization, and ball
trajectory estimation.

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
