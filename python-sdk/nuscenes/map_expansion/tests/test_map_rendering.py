import unittest
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from shapely.geometry import Polygon

from nuscenes.map_expansion.map_api import NuScenesMapExplorer


class TestMapRendering(unittest.TestCase):

    def test_polygon_layers_preserve_holes_and_labels(self):
        polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)],
                          holes=[[(1, 1), (1, 3), (3, 3), (3, 1)]])
        map_api = SimpleNamespace(
            canvas_edge=(4, 4),
            non_geometric_polygon_layers=['drivable_area', 'lane'],
            drivable_area=[{'token': 'area', 'polygon_tokens': ['polygon', 'polygon']}],
            lane=[{'token': 'lane-1', 'polygon_token': 'polygon'},
                  {'token': 'lane-2', 'polygon_token': 'polygon'}],
            extract_polygon=lambda token: polygon)
        explorer = NuScenesMapExplorer(map_api)

        for layer_name in map_api.non_geometric_polygon_layers:
            with self.subTest(layer=layer_name):
                figure = Figure(figsize=(2, 2), dpi=100)
                canvas = FigureCanvasAgg(figure)
                ax = figure.add_axes([0, 0, 1, 1])
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 4)
                ax.set_axis_off()

                explorer._render_polygon_layer(ax, layer_name, alpha=1)
                canvas.draw()
                image = np.asarray(canvas.buffer_rgba())

                # The shell is filled, while the polygon's hole stays white.
                self.assertFalse(np.all(image[25, 25, :3] == 255))
                np.testing.assert_array_equal(image[100, 100, :3], [255, 255, 255])
                self.assertEqual(len(ax.patches), 2)
                self.assertEqual(ax.patches[0].get_label(), layer_name)
                self.assertNotEqual(ax.patches[1].get_label(), layer_name)
