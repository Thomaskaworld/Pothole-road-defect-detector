import unittest

from detector import calculate_union_area_percent


class UnionCoverageTests(unittest.TestCase):
    def test_single_rectangle(self):
        self.assertEqual(
            calculate_union_area_percent([(0, 0, 10, 20)], 100, 100),
            2.0,
        )

    def test_overlapping_rectangles_are_counted_once(self):
        # Two 10x10 rectangles overlap by 5x10, so union area is 150 pixels.
        self.assertEqual(
            calculate_union_area_percent(
                [(0, 0, 10, 10), (5, 0, 10, 10)], 100, 100
            ),
            1.5,
        )

    def test_clips_rectangles_to_image(self):
        self.assertEqual(
            calculate_union_area_percent([(-5, -5, 10, 10)], 100, 100),
            0.25,
        )

    def test_empty_boxes(self):
        self.assertEqual(calculate_union_area_percent([], 100, 100), 0.0)

    def test_invalid_image_dimensions(self):
        self.assertEqual(calculate_union_area_percent([(0, 0, 10, 10)], 0, 100), 0.0)


if __name__ == "__main__":
    unittest.main()
