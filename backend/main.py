from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

class ObjectDetection(BaseModel):
    id: int
    name: str
    confidence: float
    bbox: List[int]

class SceneSchema(BaseModel):
    objects: List[ObjectDetection]

app = FastAPI(title="Scene Detection Backend API")

@app.post("/scene")
async def receive_scene(scene: SceneSchema):
    try:
        count = len(scene.objects)
        return {
            "status": "success",
            "objects_received": count
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))