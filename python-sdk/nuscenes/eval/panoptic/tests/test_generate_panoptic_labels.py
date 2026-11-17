import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.panoptic import get_panoptic_from_seg_det_or_track as generation


class TestGeneratePanopticLabels(unittest.TestCase):

    def setUp(self):
        config_factory('tracking_nips_2019')

    def test_uint8_stuff_labels_with_detection_and_tracking(self):
        labels = np.array([0, 4, 10, 11, 12, 13, 14, 15, 16], dtype=np.uint8)
        expected = np.array([0, 4001, 0, 11000, 12000, 13000, 14000, 15000, 16000], dtype=np.uint16)
        pose = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
        records = {
            ('sample', 'sample'): {'scene_token': 'scene', 'data': {'LIDAR_TOP': 'lidar'}},
            ('sample_data', 'lidar'): {'token': 'lidar', 'filename': 'points.bin',
                                     'calibrated_sensor_token': 'sensor', 'ego_pose_token': 'pose'},
            ('calibrated_sensor', 'sensor'): pose,
            ('ego_pose', 'pose'): pose
        }

        for task in ['segmentation', 'tracking']:
            with self.subTest(task=task), TemporaryDirectory() as directory:
                root = Path(directory)
                np.zeros((len(labels), 5), dtype=np.float32).tofile(root / 'points.bin')
                label_dir = root / 'lidarseg' / 'mini'
                label_dir.mkdir(parents=True)
                labels.tofile(label_dir / 'lidar_lidarseg.bin')

                box = {'sample_token': 'sample', 'translation': [0, 0, 0], 'size': [2, 2, 2],
                       'rotation': [1, 0, 0, 0], 'velocity': [0, 0]}
                if task == 'segmentation':
                    box.update(detection_name='car', detection_score=1.0, attribute_name='vehicle.parked')
                else:
                    box.update(tracking_name='car', tracking_score=1.0, tracking_id='car-1')
                predictions = root / 'predictions.json'
                predictions.write_text(json.dumps({'results': {'sample': [box]}, 'meta': {}}))

                nusc = SimpleNamespace(version='v1.0-mini', dataroot=directory,
                                       get=lambda table, token: records[table, token])
                with patch.object(generation, 'get_samples_in_eval_set', return_value=['sample']), \
                        patch.object(generation, 'PanopticClassMapper') as mapper:
                    mapper.return_value.get_coarse2idx.return_value = {'car': 4}
                    generation.generate_panoptic_labels(nusc, directory, str(predictions), 'mini_val',
                                                        task=task, out_dir=directory)

                with np.load(root / 'panoptic' / 'mini_val' / 'lidar_panoptic.npz') as result:
                    self.assertEqual(result['data'].dtype, np.uint16)
                    np.testing.assert_array_equal(result['data'], expected)
