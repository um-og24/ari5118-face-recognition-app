import sys
sys.dont_write_bytecode = True

from pydantic import BaseModel
from typing import List, Optional

class FaceRecognitionResult_Models(BaseModel):
    library: Optional[str] = None
    detector: Optional[str] = None
    model: Optional[str] = None
    boxes: Optional[List[List[int]]] = [[]]
    names: Optional[List[str]] = []
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    avg_cpu_usage: Optional[float] = None
    avg_ram_usage: Optional[float] = None
    image: Optional[str] = None


class FaceRecognitionResult(BaseModel):
    models: List[FaceRecognitionResult_Models]


class RecognitionResponse(BaseModel):
    results: FaceRecognitionResult

