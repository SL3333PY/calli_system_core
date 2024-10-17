from pydantic import BaseModel
from typing import Optional


class RecognizePostRequest(BaseModel):
    imitatedImage: Optional[str] = None


class RecognizePostResponse(BaseModel):
    text: Optional[str] = None


class EvaluatePostRequest(BaseModel):
    imitatedImage: Optional[str] = None
    generatedImage: Optional[str] = None


class EvaluatePostResponse(BaseModel):
    overlappedImage: Optional[str]
    ssim: Optional[float]
    aHash: Optional[float]
    dHash: Optional[float]
    pHash: Optional[float]
