from __future__ import annotations

from event_recorder.exclusion import (
    box_fully_inside_polygon,
    filter_detection_result_by_exclusion,
    map_display_point_to_frame,
    map_display_point_to_frame_clamped,
    nearest_polygon_vertex,
    point_in_polygon,
)
from event_recorder.models import DetectedObject, DetectionResult


def _detected(name: str, xyxy: tuple[float, float, float, float]) -> DetectedObject:
    return DetectedObject(
        class_id=0,
        class_name=name,
        confidence=0.9,
        xyxy=xyxy,
    )


def test_point_in_polygon_treats_boundary_as_inside():
    polygon = ((10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0))

    assert point_in_polygon((60.0, 60.0), polygon)
    assert point_in_polygon((10.0, 60.0), polygon)
    assert not point_in_polygon((9.0, 60.0), polygon)


def test_box_fully_inside_polygon_is_excluded_when_all_corners_are_inside():
    polygon = ((10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0))

    assert box_fully_inside_polygon((20.0, 20.0, 100.0, 100.0), polygon)
    assert box_fully_inside_polygon((10.0, 10.0, 110.0, 110.0), polygon)


def test_box_partly_outside_polygon_is_not_excluded():
    polygon = ((10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0))

    assert not box_fully_inside_polygon((20.0, 20.0, 130.0, 100.0), polygon)


def test_filter_detection_result_updates_detected_flag_when_all_are_excluded():
    polygon = ((10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0))
    result = DetectionResult(
        frame_id=1,
        captured_at_monotonic=1.0,
        detected=True,
        detections=(_detected("person", (20.0, 20.0, 100.0, 100.0)),),
    )

    filtered = filter_detection_result_by_exclusion(result, polygon)

    assert filtered.detected is False
    assert filtered.detections == ()


def test_filter_detection_result_keeps_partly_outside_detections():
    polygon = ((10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0))
    outside = _detected("car", (20.0, 20.0, 130.0, 100.0))
    inside = _detected("person", (20.0, 20.0, 100.0, 100.0))
    result = DetectionResult(
        frame_id=1,
        captured_at_monotonic=1.0,
        detected=True,
        detections=(inside, outside),
    )

    filtered = filter_detection_result_by_exclusion(result, polygon)

    assert filtered.detected is True
    assert filtered.detections == (outside,)


def test_display_point_maps_to_frame_point_with_letterboxing():
    point = map_display_point_to_frame(
        point=(400.0, 300.0),
        widget_size=(800, 600),
        frame_size=(1280, 720),
    )

    assert point == (640.0, 360.0)


def test_display_point_in_letterbox_returns_none():
    point = map_display_point_to_frame(
        point=(400.0, 20.0),
        widget_size=(800, 600),
        frame_size=(1280, 720),
    )

    assert point is None


def test_clamped_display_point_in_letterbox_maps_to_frame_edge():
    point = map_display_point_to_frame_clamped(
        point=(400.0, 20.0),
        widget_size=(800, 600),
        frame_size=(1280, 720),
    )

    assert point == (640.0, 0.0)


def test_clamped_display_point_outside_widget_maps_to_frame_corner():
    point = map_display_point_to_frame_clamped(
        point=(-50.0, 700.0),
        widget_size=(800, 600),
        frame_size=(1280, 720),
    )

    assert point == (0.0, 720.0)


def test_nearest_polygon_vertex_returns_vertex_within_radius():
    polygon = ((10.0, 10.0), (100.0, 10.0), (100.0, 100.0))

    assert nearest_polygon_vertex((96.0, 12.0), polygon, max_distance=6.0) == 1


def test_nearest_polygon_vertex_returns_none_outside_radius():
    polygon = ((10.0, 10.0), (100.0, 10.0), (100.0, 100.0))

    assert nearest_polygon_vertex((90.0, 20.0), polygon, max_distance=6.0) is None
