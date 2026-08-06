from types import SimpleNamespace

from scene_localizer.latency_trace import LatencyTracer, source_stamp_ns


class FakeNow:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds


class FakeClock:
    def __init__(self):
        self.value = 1_000_000_000

    def now(self):
        self.value += 1_000
        return FakeNow(self.value)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self):
        self.clock = FakeClock()
        self.publisher = None
        self.publisher_calls = 0

    def create_publisher(self, _message_type, _topic, _qos):
        self.publisher_calls += 1
        self.publisher = FakePublisher()
        return self.publisher

    def get_clock(self):
        return self.clock

    def get_name(self):
        return "test_node"


def make_message(sec=12, nanosec=345):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))
    )


def make_tracer(node, enabled=True):
    return LatencyTracer(
        node,
        enabled=enabled,
        topic="/intercept_trace/test",
        run_id="run",
        modality="vision",
        stage="test_stage",
    )


def test_disabled_tracing_has_no_side_effects():
    node = FakeNode()
    tracer = make_tracer(node, enabled=False)

    assert not tracer.enabled
    assert node.publisher_calls == 0
    assert tracer.begin(make_message()) is None
    tracer.mark_start(None)
    tracer.finish(None, valid=True, event="published")
    assert node.clock.value == 1_000_000_000


def test_trace_timestamps_are_monotonic_and_source_is_propagated():
    node = FakeNode()
    tracer = make_tracer(node)
    span = tracer.begin(make_message())
    tracer.mark_start(span)
    tracer.finish(span, valid=True, event="published")

    trace = node.publisher.messages[0]
    assert trace.source_stamp_ns == 12_000_000_345
    assert trace.receipt_ros_stamp_ns <= trace.start_ros_stamp_ns <= trace.end_ros_stamp_ns
    assert trace.start_steady_ns <= trace.end_steady_ns


def test_invalid_fit_still_publishes_invalid_trace():
    node = FakeNode()
    tracer = make_tracer(node)
    span = tracer.begin(make_message())
    tracer.finish(
        span,
        valid=False,
        event="rejected",
        scalar_value=float("inf"),
        detail={"reason": "insufficient_samples", "observation_count": 1},
    )

    trace = node.publisher.messages[0]
    assert trace.valid is False
    assert trace.scalar_value == 0.0
    assert '"observation_count":1' in trace.detail_json


def test_optional_message_fields_do_not_crash_trace_publication():
    node = FakeNode()
    tracer = make_tracer(node)
    span = tracer.begin(SimpleNamespace())
    tracer.finish(span, valid=False, event="rejected", detail=None)

    trace = node.publisher.messages[0]
    assert source_stamp_ns(SimpleNamespace()) == 0
    assert trace.source_stamp_ns == 0
    assert trace.detail_json == ""
