from pydantic import BaseModel
from enum import Enum
from typing import Optional

class WrapCarRequest(BaseModel):
    brightness: int = 0
    contrast: float = 1.0

class WarningType(str, Enum):
    NO_CAR = "no_car"
    NOT_SIDE_VIEW = "not_side_view"
    NONE = "none"

class CarProcessingResponse(BaseModel):
    image_url: str
    warning: Optional[WarningType] = WarningType.NONE
    warning_message: Optional[str] = None
    success: bool = True
