import unittest
from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, MagicMock
import numpy as np
from io import BytesIO
from PIL import Image

client = TestClient(app)

class TestGunDetectorAPI(unittest.TestCase):

    def test_get_model_info(self):
        response = client.get("/model_info")

        expected_response = {
            "model_name": "Gun detector",
            "gun_detector_model": "DetectionModel",
            "semantic_segmentation_model": "SegmentationModel",
            "input_type": "image"
        }

        assert response.status_code == 200

        assert response.json() == expected_response
        
    @patch('src.predictor.GunDetector.detect_guns')
    def test_detect_guns(self, mock_detect_guns):
        mock_detection = MagicMock()
        mock_detection.pred_type = "OD"
        mock_detection.n_detections = 1
        mock_detection.boxes = [[181, 340, 415, 495]]
        mock_detection.labels = ["Pistol"]
        mock_detection.confidences = [0.6560305953025818]
        mock_detect_guns.return_value = mock_detection

        with open("gun1.jpg", "rb") as img_file:
            files = {"file": img_file}
            response = client.post("/detect_guns", files=files, data={"threshold": 0.5})

        assert response.status_code == 200

        json_response = response.json()

        assert json_response["pred_type"] == "OD"
        assert json_response["n_detections"] == 1

        assert len(json_response["boxes"]) == 1

        assert len(json_response["labels"]) == 1
        assert json_response["labels"][0] == "Pistol"

        assert len(json_response["confidences"]) == 1

    @patch('src.predictor.GunDetector.segment_people')
    def test_detect_people(self, mock_segment_people):
        mock_segmentation = MagicMock()
        mock_segmentation.pred_type = "SEG"
        mock_segmentation.n_detections = 1
        mock_segmentation.polygons = [
            [
                [702, 90], [702, 96], [698, 100], [694, 100], [693, 102], 
                [689, 102], [687, 104], [685, 104], [683, 106], [669, 106],
            ]
        ]
        mock_segmentation.boxes = [[241, 89, 1126, 809]]
        mock_segmentation.labels = ["danger"]
        mock_segment_people.return_value = mock_segmentation

        with open("gun1.jpg", "rb") as img_file:
            files = {"file": img_file}        
            response = client.post("/detect_people", files=files, data={"threshold": 0.5})

        assert response.status_code == 200

        json_response = response.json()

        assert json_response["pred_type"] == "SEG"
        assert json_response["n_detections"] == 1

        assert len(json_response["polygons"]) == 1

        assert len(json_response["boxes"]) == 1

        assert len(json_response["labels"]) == 1
        assert json_response["labels"][0] == "danger"

    @patch('src.predictor.GunDetector.detect_guns')
    @patch('src.predictor.annotate_detection')
    def test_annotate_guns(self, mock_annotate_detection, mock_detect_guns):
        mock_detection = MagicMock()
        mock_detection.n_detections = 1
        mock_detect_guns.return_value = mock_detection
        
        mock_annotate_detection.return_value = np.zeros((640, 480, 3), dtype=np.uint8)

        image = Image.new('RGB', (640, 480), color = (73, 109, 137))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        response = client.post("/annotate_guns", files={"file": ("gun1.jpg", img_byte_arr, "image/jpeg")}, data={"threshold": 0.5})

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    @patch('src.predictor.GunDetector.segment_people')
    @patch('src.predictor.annotate_segmentation')
    def test_annotate_people(self, mock_annotate_segmentation, mock_segment_people):
        mock_segmentation = MagicMock()
        mock_segmentation.n_detections = 1
        mock_segment_people.return_value = mock_segmentation

        mock_annotate_segmentation.return_value = np.zeros((640, 480, 3), dtype=np.uint8)

        image = Image.new('RGB', (640, 480), color = (73, 109, 137))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        response = client.post("/annotate_people", files={"file": ("gun1.jpg", img_byte_arr, "image/jpeg")}, data={"threshold": 0.5})

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        
    
    @patch('src.predictor.GunDetector.detect_guns')
    @patch('src.predictor.GunDetector.segment_people')
    @patch('src.predictor.annotate_detection')
    @patch('src.predictor.annotate_segmentation')
    def test_annotate(self, mock_annotate_segmentation, mock_annotate_detection, mock_segment_people, mock_detect_guns):
        mock_detection = MagicMock()
        mock_detection.n_detections = 1
        mock_detect_guns.return_value = mock_detection
        
        mock_segmentation = MagicMock()
        mock_segmentation.n_detections = 1
        mock_segment_people.return_value = mock_segmentation
        
        mock_annotate_detection.return_value = np.zeros((640, 480, 3), dtype=np.uint8)
        mock_annotate_segmentation.return_value = np.zeros((640, 480, 3), dtype=np.uint8)
        
        image = Image.new('RGB', (640, 480), color=(73, 109, 137))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        response = client.post("/annotate", files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}, data={"threshold": 0.5})

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

if __name__ == "__main__":
    unittest.main()
