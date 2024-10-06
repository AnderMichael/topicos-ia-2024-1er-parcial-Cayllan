from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import (
    Detection,
    PredictionType,
    Segmentation,
    PersonType,
    Gun,
    Person,
    GunType,
    PixelLocation,
)
from src.config import get_settings

SETTINGS = get_settings()


def define_guns(detection: Detection):
    guns: list[Gun] = []
    for i, gun_box in enumerate(detection.boxes):
        gs_box = box(gun_box[0], gun_box[1], gun_box[2], gun_box[3])
        location = gs_box.centroid
        gun = Gun(
            gun_type=(
                GunType.pistol
                if detection.labels[i] == "Pistol"
                else GunType.rifle
            ),
            location=PixelLocation(x=int(location.x), y=int(location.y)),
        )
        guns.append(gun)
    return guns

def define_people(segmentation: Segmentation):
    people: list[Person] = []
    for i, person_box in enumerate(segmentation.polygons):
        polygon_segment: Polygon = Polygon(((point[0], point[1]) for point in person_box))
        location = polygon_segment.centroid
        person = Person(
            person_type=(
                PersonType.danger
                if segmentation.labels[i] == PersonType.danger
                else PersonType.safe
            ),
            location=PixelLocation(x=int(location.x), y=int(location.y)),
            area=int(polygon_segment.area)
        )
        people.append(person)
    return people

def match_gun_bbox(
    segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10
) -> list[int] | None:
    matched_box = None
    polygon_segment: Polygon = Polygon(((point[0], point[1]) for point in segment))
    gun_boxes: list[Polygon] = [
        box(bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bboxes
    ]

    for gun_box in gun_boxes:
        if gun_box.distance(polygon_segment) < max_distance:
            matched_box = gun_box

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (0, 0, 255)
    annotated_img = image_array.copy()
    for label, conf, box in zip(
        detection.labels, detection.confidences, detection.boxes
    ):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            ann_color,
            1,
        )
    return annotated_img


def annotate_segmentation(
    image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True
) -> np.ndarray:
    red_color = (255, 0, 0)
    green_color = (0, 255, 0)

    masked_img = image_array.copy()

    for area, label in zip(segmentation.polygons, segmentation.labels):
        final_color = red_color if label == PersonType.danger else green_color
        masked_img = cv2.fillPoly(
            masked_img, [np.array(area, dtype=np.int32)], final_color
        )

    annotated_img = image_array.copy()

    if draw_boxes:
        for box, label in zip(segmentation.boxes, segmentation.labels):
            final_color = red_color if label == PersonType.danger else green_color
            x1, y1, x2, y2 = box
            annotated_img = cv2.rectangle(
                annotated_img, (x1, y1), (x2, y2), final_color, 3
            )
            annotated_img = cv2.putText(
                annotated_img,
                f"{label}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                final_color,
                2,
            )

    annotated_img = cv2.addWeighted(annotated_img, 0.4, masked_img, 0.6, 0)
    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [i for i in range(len(labels)) if labels[i] in [3, 4]]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [results.names[labels[i]] for i in indexes]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )

    def segment_people(
        self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10
    ):
        persons_segments = self.seg_model(image_array, conf=threshold)[0]

        labels = persons_segments.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] == 0
        ]  # Dado que 0 representa personas
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(persons_segments.boxes.xyxy.numpy())
            if i in indexes
        ]

        polygons = []
        if persons_segments.masks:
            for mask in persons_segments.masks.xy:
                polygons.append(mask.astype(int))

        guns_boxes = self.detect_guns(image_array, threshold).boxes

        labels_txt = []

        for polygon in polygons:
            gun_matched = match_gun_bbox(polygon, guns_boxes, max_distance)
            if gun_matched:
                labels_txt.append(PersonType.danger)
            else:
                labels_txt.append(PersonType.safe)

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(boxes),
            polygons=polygons,
            boxes=boxes,
            labels=labels_txt,
        )
