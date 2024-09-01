from pydantic import BaseModel, RootModel
from typing import List, Dict, Optional, Union

class Request(BaseModel):
    ten_photos: List[int]
    tags: List[str]
    people_ids: List[int] = []
    user_relation: str

class orderQaResult(BaseModel):
    placementImgId: int
    compositionId: int
    status: int


class orderQaResponse(BaseModel):
    error: bool
    error_description: Union[str, None]
    result: List[orderQaResult]


class LogicalSelectionState(RootModel):
    root: List[str]

class PlacementImg(BaseModel):
    placementImgId: int
    compositionPackageId: int
    compositionId: int
    boxId: int
    photoId: int
    cropX: float
    cropY: float
    cropWidth: float
    cropHeight: float
    rotate: int
    projectId: int
    photoFilter: int
    photo: Optional[str] = None

class Box(BaseModel):
    id: int
    x: float
    y: float
    width: float
    height: float
    layer: int
    layerOrder: int
    type: int

class Composition(BaseModel):
    compositionId: int
    compositionPackageId: int
    designId: int
    styleId: int
    revisionCounter: int
    copies: int
    boxes: List[Box]
    logicalSelectionsState: List[str]

class Placement(BaseModel):
    accountId: int
    alerts: Optional[str] = None
    bundle: Optional[str] = None
    compositionPackageId: int
    compositions: List[Composition]
    copies: int
    countCompositions: int
    engagementId: Optional[str] = None
    externalReference: Optional[str] = None
    fulfillerId: int
    guserId: int
    logicalSelectionsState: List[str]
    packageDesignId: Optional[str] = None
    packageFinishingId: int
    packageStyleId: int
    packageTypeId: int
    placementsImg: List[PlacementImg]
    placementsTxt: List
    productId: int
    projectId: int
    revisionCounter: int
    specialOptions: Optional[str] = None
    status: int
    storeId: int
    userId: int
    userJobId: int
    __type: str

class PlacementRoot(RootModel):
    root: List[Placement]


class albumResponse(BaseModel):
    error:bool
    error_description:str
    result: PlacementRoot


if __name__ == "__main__":
    import json
    # Load your JSON data
    with open('album.json', 'r') as f:
        data = json.load(f)

    # Parse the data
    parsed_data = PlacementRoot.model_validate(data)

    # Access the data
    first_placement = parsed_data.root[0]
    print(first_placement.accountId)
    print(first_placement.compositions[0].boxes[0].id)