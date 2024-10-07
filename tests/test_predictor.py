import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from shapely.geometry import Polygon
from src.predictor import (
    define_guns,
    define_people,
    match_gun_bbox,
    annotate_detection,
    GunDetector
)
from src.models import Detection, Segmentation, GunType, PersonType, PixelLocation

class TestDefineGuns(unittest.TestCase):
    def test_define_guns(self):
        detection = Detection(
            pred_type="OD",
            n_detections=1,
            boxes=[[100, 150, 200, 250]], 
            labels=["Pistol"],
            confidences=[0.95]
        )

        guns = define_guns(detection)

        assert len(guns) == 1
        assert guns[0].gun_type == GunType.pistol
        assert guns[0].location == PixelLocation(x=150, y=200)

class TestDefinePeople(unittest.TestCase):
    def test_define_people(self):
        segmentation = Segmentation(
            pred_type="SEG",
            n_detections=1,
            polygons=[[[100, 150], [150, 200], [200, 150]]],
            boxes=[[100, 150, 200, 250]],
            labels=["danger"]
        )

        people = define_people(segmentation)

        assert len(people) == 1
        assert people[0].person_type == PersonType.danger
        assert people[0].location == PixelLocation(x=150, y=166)
        assert people[0].area == int(Polygon(segmentation.polygons[0]).area)

class TestMatchGunBbox(unittest.TestCase):
    def test_match_gun_bbox(self):
        segment = [[100, 150], [150, 200], [200, 150]]
        bboxes = [[90, 140, 210, 260]]

        matched_box = match_gun_bbox(segment, bboxes, max_distance=10)

        assert matched_box is not None

# Test Annotate Detection
class TestAnnotateDetection(unittest.TestCase):
    def test_annotate_detection(self):
        detection = Detection(
            pred_type="OD",
            n_detections=1,
            boxes=[[100, 150, 200, 250]],
            labels=["Pistol"],
            confidences=[0.95]
        )

        image_array = np.zeros((300, 300, 3), dtype=np.uint8)

        annotated_img = annotate_detection(image_array, detection)

        assert np.any(annotated_img != image_array)

class TestGunDetector(unittest.TestCase):
    @patch('src.predictor.YOLO')
    def test_detect_guns(self, mock_yolo):
        mock_od_model = MagicMock()
        mock_od_model.return_value = [
            MagicMock(
                boxes=MagicMock(
                    cls=np.array([3]), 
                    xyxy=np.array([[100, 150, 200, 250]]), 
                    conf=np.array([0.95])
                ),
                names={3: "Pistol"}
            )
        ]
        mock_yolo.return_value = mock_od_model

        gun_detector = GunDetector()

        image_array = np.zeros((640, 480, 3), dtype=np.uint8)

        detection = gun_detector.detect_guns(image_array)

        assert detection.n_detections == 1
        assert detection.labels == ["Pistol"]
        assert detection.boxes == [[100, 150, 200, 250]]
        assert detection.confidences == [0.95]

if __name__ == '__main__':
    unittest.main()

