from fastapi import FastAPI, HTTPException
import uvicorn
import logging

from app.models import (
    RecognizePostRequest,
    RecognizePostResponse,
    EvaluatePostRequest,
    EvaluatePostResponse
)
from app.services.comparison import (
    match_images,
    process_image,
    stretch_image_region
)
from app.services.image_process import (
    convert_base64_to_image,
    convert_image_to_base64,
    save_image,
)
from app.services.evaluation import get_similarity
from app.services.character_recognition import detect_characters

app = FastAPI(debug=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/recognize", response_model=RecognizePostResponse)
def recognize(body: RecognizePostRequest) -> RecognizePostResponse:
    try:
        image = convert_base64_to_image(body.imitatedImage)
        image_path = save_image(image)
        text = detect_characters(image_path)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return RecognizePostResponse(text=text)


@app.post("/evaluate", response_model=EvaluatePostResponse)
def evaluate(body: EvaluatePostRequest) -> EvaluatePostResponse:
    try:
        imitated_image = convert_base64_to_image(body.imitatedImage)    # grayscale image
        generated_image = convert_base64_to_image(body.generatedImage)
        target_size = 256

        imitated_image, region_rect1 = process_image(imitated_image, target_size)
        generated_image, region_rect2 = process_image(generated_image, target_size)
        stretched_imitated_image = stretch_image_region(
            imitated_image, generated_image, region_rect1, region_rect2, target_size, target_size)

        overlapped_image = match_images(stretched_imitated_image, generated_image)
        base64_image = "data:image/png;base64," + convert_image_to_base64(overlapped_image)

        ssim, aHash, dHash, pHash = get_similarity(stretched_imitated_image, generated_image)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return EvaluatePostResponse(overlappedImage=base64_image, ssim=ssim, aHash=aHash, dHash=dHash, pHash=pHash)


if __name__ == '__main__':
    uvicorn.run(app, port=9000)

