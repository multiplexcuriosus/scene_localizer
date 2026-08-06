"""Opt-in raw latency tracing shared by scene-localizer pipeline nodes."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from intercept_latency_monitor.msg import LatencyTrace
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


@dataclass
class TraceSpan:
    sequence: int
    source_stamp_ns: int
    receipt_ros_stamp_ns: int
    start_ros_stamp_ns: int
    start_steady_ns: int


def source_stamp_ns(message: Any) -> int:
    """Return a header timestamp in nanoseconds, or zero when unavailable."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    try:
        sec = int(getattr(stamp, "sec", 0))
        nanosec = int(getattr(stamp, "nanosec", 0))
    except (TypeError, ValueError):
        return 0
    stamp_ns = sec * 1_000_000_000 + nanosec
    return stamp_ns if stamp_ns > 0 else 0


class LatencyTracer:
    """Publish one raw LatencyTrace for each completed callback span."""

    def __init__(
        self,
        node: Any,
        *,
        enabled: bool,
        topic: str,
        run_id: str,
        modality: str,
        stage: str,
    ) -> None:
        self._node = node
        self._run_id = run_id
        self._modality = modality
        self._stage = stage
        self._sequence = 0
        self._publisher = None
        if enabled:
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._publisher = node.create_publisher(LatencyTrace, topic, qos)

    @property
    def enabled(self) -> bool:
        return self._publisher is not None

    def begin(self, message: Any) -> Optional[TraceSpan]:
        if self._publisher is None:
            return None
        self._sequence += 1
        receipt_ros_stamp_ns = int(self._node.get_clock().now().nanoseconds)
        return TraceSpan(
            sequence=self._sequence,
            source_stamp_ns=source_stamp_ns(message),
            receipt_ros_stamp_ns=receipt_ros_stamp_ns,
            start_ros_stamp_ns=int(self._node.get_clock().now().nanoseconds),
            start_steady_ns=time.monotonic_ns(),
        )

    def mark_start(self, span: Optional[TraceSpan]) -> None:
        """Move the measured stage start to the current instant."""
        if self._publisher is None or span is None:
            return
        span.start_ros_stamp_ns = int(self._node.get_clock().now().nanoseconds)
        span.start_steady_ns = time.monotonic_ns()

    def finish(
        self,
        span: Optional[TraceSpan],
        *,
        valid: bool,
        event: str,
        scalar_value: Optional[float] = None,
        detail: Optional[Dict[str, Any]] = None,
        end_ros_stamp_ns: Optional[int] = None,
        end_steady_ns: Optional[int] = None,
    ) -> None:
        if self._publisher is None or span is None:
            return

        trace = LatencyTrace()
        trace.run_id = self._run_id
        trace.stage = self._stage
        trace.event = event
        trace.modality = self._modality
        trace.node_name = self._node.get_name()
        trace.sequence = span.sequence
        trace.parent_sequence = 0
        trace.source_stamp_ns = span.source_stamp_ns
        trace.receipt_ros_stamp_ns = span.receipt_ros_stamp_ns
        trace.start_ros_stamp_ns = span.start_ros_stamp_ns
        trace.end_ros_stamp_ns = (
            int(end_ros_stamp_ns)
            if end_ros_stamp_ns is not None
            else int(self._node.get_clock().now().nanoseconds)
        )
        trace.start_steady_ns = span.start_steady_ns
        trace.end_steady_ns = (
            int(end_steady_ns) if end_steady_ns is not None else time.monotonic_ns()
        )
        trace.valid = bool(valid)
        value = float(scalar_value) if scalar_value is not None else 0.0
        trace.scalar_value = value if math.isfinite(value) else 0.0
        trace.detail_json = (
            json.dumps(detail, separators=(",", ":"), sort_keys=True)
            if detail
            else ""
        )
        self._publisher.publish(trace)
