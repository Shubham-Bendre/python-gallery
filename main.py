from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
from typing import List
import json
from pydantic import BaseModel, Field
from datetime import datetime
import os
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB configuration
MONGO_URL = ""
client = AsyncIOMotorClient(MONGO_URL)
db = client.face_recognition_db

# Cloudinary configuration
cloudinary.config(
    cloud_name="",
    api_key="",
    api_secret=""
)
# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


# Pydantic models
class ImageModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    cloudinary_url: str
    filename: str
    uploaded_at: datetime
    faces: List[PyObjectId]

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True

class PersonModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    reference_face_features: List[float]
    face_count: int
    created_at: datetime

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True

# Routes
@app.get("/api/images", response_model=List[ImageModel])
async def get_all_images():
    """Get all images from the database"""
    images = await db.images.find().to_list(1000)
    return images

@app.get("/api/images/{image_id}", response_model=ImageModel)
async def get_image(image_id: str):
    """Get a specific image by ID"""
    if not ObjectId.is_valid(image_id):
        raise HTTPException(status_code=400, detail="Invalid image ID")
    
    image = await db.images.find_one({"_id": ObjectId(image_id)})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image

@app.get("/api/people", response_model=List[PersonModel])
async def get_all_people():
    """Get all people from the database"""
    people = await db.people.find().to_list(1000)
    return people

@app.get("/api/people/{person_id}/photos", response_model=List[ImageModel])
async def get_person_photos(person_id: str):
    """Get all photos containing a specific person"""
    if not ObjectId.is_valid(person_id):
        raise HTTPException(status_code=400, detail="Invalid person ID")
    
    # Find all faces for this person
    faces = await db.faces.find({"person_id": ObjectId(person_id)}).to_list(1000)
    image_ids = [face["image_id"] for face in faces]
    
    # Get the corresponding images
    images = await db.images.find({"_id": {"$in": image_ids}}).to_list(1000)
    return images

@app.post("/api/search")
async def search_similar_faces(file: UploadFile = File(...)):
    """Search for similar faces using an uploaded image"""
    try:
        # Read and process the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return {"message": "No faces detected in the image", "results": []}
        
        # Get features for the first detected face
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        search_features = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        search_features = cv2.normalize(search_features, search_features).flatten()
        
        # Search for similar faces in the database
        similar_faces = []
        async for face in db.faces.find():
            stored_features = np.array(face['features'])
            correlation = np.corrcoef(search_features, stored_features)[0, 1]
            
            if correlation > 0.7:  # Similarity threshold
                image = await db.images.find_one({"_id": face["image_id"]})
                if image:
                    similar_faces.append({
                        "_id": str(image["_id"]),
                        "cloudinary_url": image["cloudinary_url"],
                        "similarity": float(correlation)
                    })
        
        # Sort by similarity
        similar_faces.sort(key=lambda x: x["similarity"], reverse=True)
        return {"results": similar_faces[:10]}  # Return top 10 matches
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File upload route for new images
@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a new image and process faces"""
    try:
        # Read the file
        contents = await file.read()
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(contents)
        cloudinary_url = upload_result["secure_url"]
        
        # Save to MongoDB
        image_doc = {
            "cloudinary_url": cloudinary_url,
            "filename": file.filename,
            "uploaded_at": datetime.now(),
            "faces": []
        }
        
        result = await db.images.insert_one(image_doc)
        
        # Process faces (using the existing face processing logic)
        # ... (face processing code here)
        
        return {"message": "Image uploaded successfully", "image_id": str(result.inserted_id)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
