import numpy as np

from src.document_pipeline.geometry import order_corners, warp_perspective


def test_order_corners_returns_clockwise():
    points = np.array([[50, 10], [10, 10], [10, 40], [50, 40]], dtype=np.float32)
    ordered = order_corners(points)
    # Esperamos orden: top-left, top-right, bottom-right, bottom-left
    np.testing.assert_array_equal(ordered[0], [10, 10])
    np.testing.assert_array_equal(ordered[1], [50, 10])
    np.testing.assert_array_equal(ordered[2], [50, 40])
    np.testing.assert_array_equal(ordered[3], [10, 40])


def test_warp_perspective_produces_expected_shape():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    contour = np.array([[10, 10], [70, 10], [70, 40], [10, 40]], dtype=np.float32)
    warped = warp_perspective(image, contour)
    # max_width = 60, max_height = 30 según las distancias del contorno
    assert warped.shape[:2] == (30, 60)
