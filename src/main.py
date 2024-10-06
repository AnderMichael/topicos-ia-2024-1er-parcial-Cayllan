import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import (
    GunDetector,
    Detection,
    Segmentation,
    GeneralDetect,
    annotate_detection,
    annotate_segmentation,
    define_guns,
    define_people
)
from src.config import get_settings
from src.models import GeneralDetect, Gun, Person

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(
    detector: GunDetector, file, threshold
) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Image format not suported for detection",
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array


def segmentation_uploadfile(
    detector: GunDetector, file, threshold
) -> tuple[Segmentation, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Image format not suported for segmentation",
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.segment_people(img_array, threshold), img_array


def detect_segmentation_uploadfile(
    detector: GunDetector, file, threshold
) -> tuple[Detection, Segmentation, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Image format not suported for segmentation",
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return (
        detector.detect_guns(img_array, threshold),
        detector.segment_people(img_array, threshold),
        img_array,
    )


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)
    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    segmentation, _ = segmentation_uploadfile(detector, file, threshold)
    return segmentation


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    segmentation, image_array = segmentation_uploadfile(detector, file, threshold)
    annotated_img = annotate_segmentation(image_array, segmentation)
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect")
def detect(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> GeneralDetect:
    detection, segmentation, _ = detect_segmentation_uploadfile(
        detector, file, threshold
    )
    general_detection = GeneralDetect(detection=detection, segmentation=segmentation)
    return general_detection


@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, segmentation, image_array = detect_segmentation_uploadfile(
        detector, file, threshold
    )
    annotated_img = annotate_detection(image_array, detection)
    annotated_img = annotate_segmentation(annotated_img, segmentation)
    # guns = define_guns(detection)
    # for gun in guns:
    #     annotated_img = cv2.circle(annotated_img, (gun.location.x, gun.location.y), 10, (0, 0, 255), 2) 
    # people = define_people(segmentation)
    # for person in people:
    #     annotated_img = cv2.circle(annotated_img, (person.location.x, person.location.y), 10, (0, 0, 255), 2) 
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/guns")
def guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:
    detection, _ = detect_uploadfile(detector, file, threshold) 
    guns = define_guns(detection)
    return guns


@app.post("/people")
def people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:
    segmentation, _ = segmentation_uploadfile(detector, file, threshold)
    people = define_people(segmentation)
    return people



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
