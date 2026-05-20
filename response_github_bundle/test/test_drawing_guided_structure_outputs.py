from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_drawing_guided_page_structure as drawing_guided
import run_region_type_classifier as type_classifier
import run_structure_visualizer as visualizer
import run_x_coordinate_region_structure as x_structure


class DrawingGuidedStructureOutputTests(unittest.TestCase):
    def test_coordinate_export_writes_english_named_folder(self) -> None:
        drawings = [
            {
                "index": 1,
                "bbox": drawing_guided.Rect(10.0, 20.0, 180.0, 20.0),
                "type": "s",
                "color": (0.961, 0.51, 0.125),
                "fill": None,
                "line_width": 0.72,
            },
            {
                "index": 2,
                "bbox": drawing_guided.Rect(30.0, 40.0, 90.0, 100.0),
                "type": "f",
                "color": None,
                "fill": (0.75, 0.75, 0.75),
                "line_width": None,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = drawing_guided.export_drawing_line_fill_coordinates(Path(temp_dir), 4, drawings)
            coordinate_dir = Path(temp_dir) / "drawing_lines_fill_coordinates"
            json_path = coordinate_dir / "page_04_drawing_lines_fill_coordinates.json"
            md_path = coordinate_dir / "page_04_drawing_lines_fill_coordinates.md"

            self.assertEqual(Path(outputs["json"]), json_path)
            self.assertEqual(Path(outputs["markdown"]), md_path)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["page_number"], 4)
            self.assertEqual(payload["drawing_lines"][0]["x0"], 10.0)
            self.assertEqual(payload["drawing_lines"][0]["width"], 170.0)
            self.assertEqual(payload["fill_coordinates"][0]["fill"], [0.75, 0.75, 0.75])
            self.assertIn("Drawing Lines", md_path.read_text(encoding="utf-8"))
            self.assertIn("Fill Coordinates", md_path.read_text(encoding="utf-8"))

    def test_visualizer_html_contains_region_boxes_and_assigned_text(self) -> None:
        pages = [
            {
                "page_number": 4,
                "page_size": [780.0, 540.0],
                "strategy": "test strategy",
                "image_file": "page_04_render.png",
                "json_file": "page_04_drawing_guided_structure.json",
                "regions": [
                    {
                        "id": "P4-R1",
                        "type": "connected_pretax_net_income",
                        "bbox": [32.0, 92.0, 263.5, 183.5],
                        "source": "unit-test",
                        "token_count": 2,
                        "markdown": "4,472\n3,438",
                        "evidence": {"line": "orange"},
                    }
                ],
            }
        ]

        html = visualizer.build_html(pages)

        self.assertIn("Mirae Asset Q3 Region Viewer", html)
        self.assertIn("Page ${page.page_number}", html)
        self.assertIn('"P4-R1"', html)
        self.assertIn("4,472", html)
        self.assertIn("regionBox", html)

    def test_region_type_classifier_scores_core_types(self) -> None:
        chart_region = {
            "id": "P16-R1",
            "bbox": [42.0, 136.0, 382.0, 300.0],
            "token_count": 28,
            "markdown": "Legend\n2021 2022 2023 2024\n31% 31% 53% 40%\n3,622 2,101 1,720 3,670",
            "evidence": {"bar_fills": [{"bbox": [1, 2, 3, 4]} for _ in range(8)]},
        }
        table_region = {
            "id": "P16-R3",
            "bbox": [42.0, 309.0, 382.0, 447.0],
            "token_count": 64,
            "markdown": "| year | 2021 | 2022 |\n| --- | --- | --- |\n| total | 1 | 2 |",
            "evidence": {"cell_fill_count": 25},
        }
        card_region = {
            "id": "P4-R6",
            "bbox": [34.0, 319.0, 381.5, 416.5],
            "token_count": 29,
            "markdown": "(connected) quarterly ROE achieved\n- first bullet sentence\n- second bullet sentence",
            "evidence": {"matching_fills": [{"bbox": [34, 319, 381, 416]}]},
        }
        roe_chart_region = {
            "id": "P4-R4",
            "bbox": [33.5, 185.0, 263.8, 276.5],
            "token_count": 19,
            "markdown": "3Q25 connected ROE\n10.9% 10.8% 10.8\n7.8% 7.9% 8.5%\n3Q24 3Q25 QoQ -0.1%p",
            "evidence": {"matching_fills": [{"bbox": [1, 2, 3, 4]} for _ in range(6)]},
        }

        for region in (chart_region, table_region, card_region, roe_chart_region):
            features = type_classifier.extract_region_features(region, 16)
            scores = type_classifier.score_region_type(features)
            region["type_classification"] = {
                "predicted_type": max(scores, key=scores.get),
                "scores": scores,
            }

        self.assertEqual(chart_region["type_classification"]["predicted_type"], "chart")
        self.assertEqual(table_region["type_classification"]["predicted_type"], "table")
        self.assertEqual(card_region["type_classification"]["predicted_type"], "highlight_card")
        self.assertNotEqual(roe_chart_region["type_classification"]["predicted_type"], "notes")

    def test_x_coordinate_structure_uses_axis_anchors(self) -> None:
        class Box:
            def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
                self.x0 = x0
                self.y0 = y0
                self.x1 = x1
                self.y1 = y1

            @property
            def cx(self) -> float:
                return (self.x0 + self.x1) / 2

            @property
            def cy(self) -> float:
                return (self.y0 + self.y1) / 2

        class Token:
            def __init__(self, text: str, x: float, y: float) -> None:
                self.text = text
                self.box = Box(x, y, x + 10, y + 8)

        tokens = [
            Token("31%", 45, 150),
            Token("3,622", 45, 180),
            Token("2021", 45, 230),
            Token("40%", 145, 150),
            Token("2,101", 145, 180),
            Token("2022", 145, 230),
        ]

        result = x_structure.build_x_coordinate_structure(tokens, "chart")

        anchors = [column["anchor"] for column in result["columns"]]
        self.assertIn("2021", anchors)
        self.assertIn("2022", anchors)
        first_column = result["columns"][0]["joined_text"]
        self.assertIn("31%", first_column)
        self.assertIn("3,622", first_column)


if __name__ == "__main__":
    unittest.main()
