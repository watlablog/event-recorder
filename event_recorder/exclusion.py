from __future__ import annotations

from event_recorder.models import DetectedObject, DetectionResult

Point = tuple[float, float]
Polygon = tuple[Point, ...]
Size = tuple[int, int]

_EPSILON = 1e-9


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if _point_on_segment(point, previous, current):
            return True
        if (y1 > y) != (y2 > y):
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x_intersection >= x:
                inside = not inside
        previous = current
    return inside


def box_fully_inside_polygon(
    xyxy: tuple[float, float, float, float], polygon: Polygon
) -> bool:
    if len(polygon) < 3:
        return False
    x1, y1, x2, y2 = xyxy
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    corners = (
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    )
    return all(point_in_polygon(corner, polygon) for corner in corners)


def filter_detections_by_exclusion(
    detections: tuple[DetectedObject, ...],
    polygon: Polygon,
) -> tuple[DetectedObject, ...]:
    if len(polygon) < 3:
        return detections
    return tuple(
        detected
        for detected in detections
        if not box_fully_inside_polygon(detected.xyxy, polygon)
    )


def filter_detection_result_by_exclusion(
    result: DetectionResult,
    polygon: Polygon,
) -> DetectionResult:
    detections = filter_detections_by_exclusion(result.detections, polygon)
    if detections == result.detections:
        return result
    return DetectionResult(
        frame_id=result.frame_id,
        captured_at_monotonic=result.captured_at_monotonic,
        detected=bool(detections),
        detections=detections,
    )


def map_display_point_to_frame(
    point: Point,
    widget_size: Size,
    frame_size: Size,
) -> Point | None:
    mapped = _map_display_point_to_frame(point, widget_size, frame_size)
    if mapped is None:
        return None
    frame_x, frame_y, frame_width, frame_height, is_inside_display = mapped
    if not is_inside_display:
        return None
    return (
        min(max(frame_x, 0.0), float(frame_width)),
        min(max(frame_y, 0.0), float(frame_height)),
    )


def map_display_point_to_frame_clamped(
    point: Point,
    widget_size: Size,
    frame_size: Size,
) -> Point | None:
    mapped = _map_display_point_to_frame(point, widget_size, frame_size)
    if mapped is None:
        return None
    frame_x, frame_y, frame_width, frame_height, _is_inside_display = mapped
    return (
        min(max(frame_x, 0.0), float(frame_width)),
        min(max(frame_y, 0.0), float(frame_height)),
    )


def _map_display_point_to_frame(
    point: Point,
    widget_size: Size,
    frame_size: Size,
) -> tuple[float, float, int, int, bool] | None:
    widget_width, widget_height = widget_size
    frame_width, frame_height = frame_size
    if widget_width <= 0 or widget_height <= 0 or frame_width <= 0 or frame_height <= 0:
        return None

    scale = min(widget_width / frame_width, widget_height / frame_height)
    display_width = frame_width * scale
    display_height = frame_height * scale
    offset_x = (widget_width - display_width) / 2.0
    offset_y = (widget_height - display_height) / 2.0
    x, y = point
    is_inside_display = (
        offset_x <= x <= offset_x + display_width
        and offset_y <= y <= offset_y + display_height
    )

    frame_x = (x - offset_x) / scale
    frame_y = (y - offset_y) / scale
    return frame_x, frame_y, frame_width, frame_height, is_inside_display


def nearest_polygon_vertex(
    point: Point,
    polygon: Polygon,
    max_distance: float,
) -> int | None:
    if max_distance < 0:
        return None
    x, y = point
    max_distance_squared = max_distance * max_distance
    closest_index: int | None = None
    closest_distance_squared = max_distance_squared
    for index, (vertex_x, vertex_y) in enumerate(polygon):
        distance_squared = (x - vertex_x) ** 2 + (y - vertex_y) ** 2
        if distance_squared <= closest_distance_squared:
            closest_index = index
            closest_distance_squared = distance_squared
    return closest_index


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > _EPSILON:
        return False
    return (
        min(x1, x2) - _EPSILON <= x <= max(x1, x2) + _EPSILON
        and min(y1, y2) - _EPSILON <= y <= max(y1, y2) + _EPSILON
    )
