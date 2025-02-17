from pydantic import BaseModel, Field
from typing import List, Optional

# Define response model
class AlbumResponse(BaseModel):
    error: bool
    error_description: Optional[str] = None
    result: Optional[dict]

class AlbumRequest(BaseModel):
    storeId: int
    accountId: int
    projectId: int
    base_url: str
    photoId: List[int] = Field(..., min_items=1, description="List of photo IDs")
    peopleIds: List[int] = Field(..., min_items=1, description="List of people IDs")
    tags: List[str] = Field(..., min_items=1, description="List of tags")
    relation: str = Field(..., description="Relation type (e.g., 'bride and groom', 'spouse', 'children')")
    category: int