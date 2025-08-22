from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Depends, Request, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timedelta
import pandas as pd
import cv2
import numpy as np
import jwt
from passlib.context import CryptContext
import math  # for BMI math

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# ------------------------------------------------------------------------------
# MongoDB (OPTIONAL) – won’t crash if env vars are missing
# ------------------------------------------------------------------------------
MONGO_URL = os.environ.get('MONGO_URL')
DB_NAME = os.environ.get('DB_NAME')
if MONGO_URL and DB_NAME:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
else:
    client = None
    db = None

# ------------------------------------------------------------------------------
# Security / App / Router
# ------------------------------------------------------------------------------
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'hjdkfh2!2jnkdjbfxjnkdc')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI()
api_router = APIRouter(prefix="/api")

# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
try:
    styles_df = pd.read_csv(ROOT_DIR / "styles.csv")
    print(f"✅ Loaded {len(styles_df)} fashion items")
except Exception as e:
    print(f"⚠️ Could not load styles.csv: {e}")
    styles_df = pd.DataFrame()

FASHION_IMAGES = {
    "Shirts": "https://via.placeholder.com/400x500/4A90E2/FFFFFF?text=Shirt",
    "Jeans": "https://via.placeholder.com/400x500/2E86AB/FFFFFF?text=Jeans",
    "T-shirts": "https://via.placeholder.com/400x500/F24236/FFFFFF?text=T-Shirt",
    "Casual Shirts": "https://via.placeholder.com/400x500/7B68EE/FFFFFF?text=Casual+Shirt",
    "Dresses": "https://via.placeholder.com/400x500/FF69B4/FFFFFF?text=Dress",
    "Track Pants": "https://via.placeholder.com/400x500/32CD32/FFFFFF?text=Track+Pants",
    "Casual Shoes": "https://via.placeholder.com/400x500/8B4513/FFFFFF?text=Shoes",
    "Handbags": "https://via.placeholder.com/400x500/DA70D6/FFFFFF?text=Handbag",
    "Watches": "https://via.placeholder.com/400x500/FFD700/000000?text=Watch",
    "Heels": "https://via.placeholder.com/400x500/DC143C/FFFFFF?text=Heels",
    "Leather Belts": "https://via.placeholder.com/400x500/654321/FFFFFF?text=Belt",
    "Sneakers": "https://via.placeholder.com/400x500/00CED1/FFFFFF?text=Sneakers",
    "Blazers": "https://via.placeholder.com/400x500/191970/FFFFFF?text=Blazer",
    "Tops": "https://via.placeholder.com/400x500/FF1493/FFFFFF?text=Top",
    "Blouses": "https://via.placeholder.com/400x500/9370DB/FFFFFF?text=Blouse",
    "Polo": "https://via.placeholder.com/400x500/228B22/FFFFFF?text=Polo",
    "Hoodies": "https://via.placeholder.com/400x500/696969/FFFFFF?text=Hoodie",
    "Skirts": "https://via.placeholder.com/400x500/FF6347/FFFFFF?text=Skirt",
    "Shorts": "https://via.placeholder.com/400x500/20B2AA/FFFFFF?text=Shorts",
    "Sweaters": "https://via.placeholder.com/400x500/9932CC/FFFFFF?text=Sweater",
    "Chinos": "https://via.placeholder.com/400x500/D2691E/FFFFFF?text=Chinos",
    "Tank Tops": "https://via.placeholder.com/400x500/FF4500/FFFFFF?text=Tank+Top",
    "Henley": "https://via.placeholder.com/400x500/4682B4/FFFFFF?text=Henley",
    "Flats": "https://via.placeholder.com/400x500/DDA0DD/FFFFFF?text=Flats",
    "Sports Watches": "https://via.placeholder.com/400x500/000000/FFFFFF?text=Sports+Watch",
    "Leggings": "https://via.placeholder.com/400x500/800080/FFFFFF?text=Leggings",
    "Cardigans": "https://via.placeholder.com/400x500/B0C4DE/000000?text=Cardigan",
    "Jackets": "https://via.placeholder.com/400x500/8B0000/FFFFFF?text=Jacket",
    "default": "https://via.placeholder.com/400x500/708090/FFFFFF?text=Fashion+Item"
}
def get_item_image(article_type: str) -> str:
    return FASHION_IMAGES.get(article_type, FASHION_IMAGES["default"])

# ------------------------------------------------------------------------------
# Skin-tone helpers
# ------------------------------------------------------------------------------
def remove_shadows_and_enhance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    denoised = cv2.bilateralFilter(enhanced_rgb, 9, 75, 75)
    return denoised

def detect_skin_region_advanced(face_img):
    h, w = face_img.shape[:2]
    ycbcr = cv2.cvtColor(face_img, cv2.COLOR_RGB2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask1 = cv2.inRange(ycbcr, lower_skin, upper_skin)

    hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
    lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask2 = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)

    r, g, b = cv2.split(face_img)
    rgb_mask = ((r > 95) & (g > 40) & (b > 20) &
                ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) &
                (np.abs(r.astype(int) - g.astype(int)) > 15) &
                (r > g) & (r > b)).astype(np.uint8) * 255

    combined_mask = cv2.bitwise_and(skin_mask1, skin_mask2)
    combined_mask = cv2.bitwise_and(combined_mask, rgb_mask)

    eye_region_mask = np.ones_like(combined_mask)
    eye_region_mask[int(h * 0.25):int(h * 0.55), :] = 0
    mouth_region_mask = np.ones_like(combined_mask)
    mouth_region_mask[int(h * 0.75):, int(w * 0.25):int(w * 0.75)] = 0

    skin_mask_clean = cv2.bitwise_and(combined_mask, eye_region_mask)
    skin_mask_clean = cv2.bitwise_and(skin_mask_clean, mouth_region_mask)

    cheek_mask = np.zeros_like(combined_mask)
    cheek_mask[int(h * 0.4):int(h * 0.7), int(w * 0.1):int(w * 0.4)] = 255
    cheek_mask[int(h * 0.4):int(h * 0.7), int(w * 0.6):int(w * 0.9)] = 255
    cheek_mask[int(h * 0.2):int(h * 0.4), int(w * 0.3):int(w * 0.7)] = 255

    final_mask = cv2.bitwise_and(skin_mask_clean, cheek_mask)
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    return final_mask

def analyze_skin_tone_advanced(face_img, skin_mask):
    skin_pixels = face_img[skin_mask > 0]
    if len(skin_pixels) < 100:
        h, w = face_img.shape[:2]
        center_region = face_img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        skin_pixels = center_region.reshape(-1, 3)

    def remove_outliers(data):
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = np.all((data >= lower) & (data <= upper), axis=1)
        return data[mask]

    cleaned = remove_outliers(skin_pixels)
    if len(cleaned) > 50:
        return np.median(cleaned, axis=0).astype(int)
    return np.mean(skin_pixels, axis=0).astype(int)

def classify_skin_tone_detailed(rgb_color):
    r, g, b = rgb_color
    brightness = (r + g + b) / 3
    red_ratio = r / max(g + b, 1)
    yellow_ratio = (r + g) / max(2 * b, 1)
    if red_ratio > 1.1 or yellow_ratio > 1.2:
        undertone = "warm"
    elif b > max(r, g):
        undertone = "cool"
    else:
        undertone = "neutral"

    if brightness > 200:
        depth = "Very Fair"
        colors = ["Cream", "Peach", "Coral", "Light Yellow", "Gold", "Warm White"] if undertone != "cool" else \
                 ["Pastels", "White", "Lavender", "Light Blue", "Pink", "Silver"]
    elif brightness > 170:
        depth = "Fair"
        colors = ["Warm Pink", "Coral", "Orange", "Yellow", "Camel", "Warm Brown"] if undertone != "cool" else \
                 ["Rose", "Berry", "Emerald", "Navy", "Purple", "Cool Gray"]
    elif brightness > 140:
        depth = "Light-Medium"
        colors = ["Rust", "Olive", "Warm Red", "Orange", "Gold", "Chocolate"] if undertone != "cool" else \
                 ["Teal", "Sapphire", "Magenta", "Cool Red", "Black", "White"]
    elif brightness > 110:
        depth = "Medium"
        colors = ["Burnt Orange", "Olive", "Warm Green", "Burgundy", "Gold", "Brown"] if undertone != "cool" else \
                 ["Royal Blue", "Purple", "Pink", "Cool Green", "Black", "Gray"]
    elif brightness > 80:
        depth = "Medium-Deep"
        colors = ["Earth Tones", "Rust", "Orange", "Yellow", "Burgundy", "Camel"] if undertone != "cool" else \
                 ["Jewel Tones", "Purple", "Blue", "Pink", "Black", "White"]
    else:
        depth = "Deep"
        colors = ["Rich Colors", "Orange", "Red", "Yellow", "Gold", "Copper"] if undertone != "cool" else \
                 ["Bright Colors", "Purple", "Blue", "Pink", "White", "Silver"]
    return f"{depth} ({undertone})", colors

def detect_skin_tone_advanced(face_img):
    enhanced_img = remove_shadows_and_enhance(face_img)
    skin_mask = detect_skin_region_advanced(enhanced_img)
    skin_color = analyze_skin_tone_advanced(enhanced_img, skin_mask)
    return skin_color

def detect_skin_tone(image_bytes: bytes) -> tuple:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format. Please use JPG, PNG, or GIF.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray_eq, 1.1, 5, minSize=(50, 50), maxSize=(500, 500))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(80, 80))
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected. Please use a clear, well-lit photo.")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    px, py = int(w * 0.05), int(h * 0.05)
    x1 = max(0, x + px); y1 = max(0, y + py)
    x2 = min(img.shape[1], x + w - px); y2 = min(img.shape[0], y + h - py)
    face_img = img[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    final_color = detect_skin_tone_advanced(face_rgb)
    hex_color = "#{:02x}{:02x}{:02x}".format(*final_color)
    skin_description, recommended_colors = classify_skin_tone_detailed(final_color)
    print(f"Advanced skin tone analysis: {skin_description}, RGB: {final_color}, Hex: {hex_color}")
    return hex_color, recommended_colors

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class SkinToneAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    detected_skin_tone: str
    recommended_colors: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class OutfitRecommendation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_id: str
    product_name: str
    category: str
    sub_category: str
    article_type: str
    base_colour: str
    gender: str
    season: str
    usage: str

class Favorite(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    item_id: str
    product_name: str
    base_colour: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# --- BodyType models ---
class BodyTypeRequest(BaseModel):
    height: str  # meters
    weight: str  # kg
    bust: str    # inches
    cup: str     # e.g. "B"
    waist: str   # inches
    hip: str     # inches

class BodyTypeResponse(BaseModel):
    bmi: int | str
    breast_multiplier: int
    breast_desc: str
    butt_desc: str
    body_shape: str
    body_type: str

# ------------------------------------------------------------------------------
# Auth helpers (work only if db is configured)
# ------------------------------------------------------------------------------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = await db.users.find_one({"email": email})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return User(**user)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# ------------------------------------------------------------------------------
# BodyType helpers (pure functions; shared logic)
# ------------------------------------------------------------------------------
def _get_bmi(weight, height):
    try:
        if not (weight and height):
            return "Error!"
        return math.floor(int(float(weight)) / (float(height) * float(height)))
    except Exception:
        return "Error!"

def _get_breast_multiplier(bust, cup):
    try:
        bust_i = int(float(bust))
        bust_scale = 'Below Average' if bust_i < 34 else 'Above Average'
    except Exception:
        bust_scale = 'Error!'
    cup = (cup or '').upper()
    if cup == '' and bust_scale == 'Error!': return 99
    elif cup in ['AA','A'] and bust_scale == 'Below Average': return 1
    elif cup in ['AA','A'] and bust_scale == 'Above Average': return 2
    elif cup in ['B','C'] and bust_scale == 'Below Average': return 2
    elif cup in ['D'] and bust_scale == 'Below Average': return 3
    elif cup in ['B','C','D'] and bust_scale == 'Above Average': return 3
    elif cup in ['DD','DDD','E','EE','EEE','F','FF','G'] and bust_scale == 'Below Average': return 3
    elif cup in ['DD','DDD','E','EE','EEE','F','FF','G'] and bust_scale == 'Above Average': return 4
    elif cup in ['FFF','GG','GGG','H','HH','I'] and bust_scale == 'Below Average': return 4
    elif cup in ['FFF','GG','GGG','H','HH','I'] and bust_scale == 'Above Average': return 5
    elif cup in ['HHH','II','III','J','JJ','K'] and bust_scale == 'Below Average': return 5
    elif cup in ['HHH','II','III','J','JJ','K'] and bust_scale == 'Above Average': return 6
    else: return 0

def _get_breast_desc(mult):
    return {1:'Tiny',2:'Small',3:'Medium',4:'Large',5:'Huge',6:'Massive',99:'Error!',0:'Error!'}.get(mult,'Error!')

def _get_butt_desc(hip):
    try:
        hip_i = int(float(hip))
        if hip_i <= 32: return 'Small'
        if 33 <= hip_i <= 39: return 'Medium'
        if 40 <= hip_i <= 43: return 'Large'
        if 44 <= hip_i <= 47: return 'Huge'
        if hip_i >= 48: return 'Massive'
    except Exception:
        pass
    return 'Error!'

def _get_body_shape(bust, waist, hip):
    try:
        bust_i = int(float(bust)); waist_i = int(float(waist)); hip_i = int(float(hip))
        if (waist_i * 1.25) <= bust_i and (waist_i * 1.25) <= hip_i: return 'Hourglass'
        elif hip_i > (bust_i * 1.05): return 'Pear'
        elif hip_i < (bust_i / 1.05): return 'Apple'
        else:
            return 'Banana' if (max(bust_i, waist_i, hip_i) - min(bust_i, waist_i, hip_i)) <= 5 else 'Banana'
    except Exception:
        return 'Error!'

def _get_body_type(index, shape):
    try:
        index_i = int(index)
        if   1 <= index_i <= 17: t='A'
        elif 18 <= index_i <= 22: t='B'
        elif 23 <= index_i <= 28: t='C'
        elif 29 <= index_i <= 54: t='D'
        else: t='E'
        if shape == 'Error!': return 'Error!'
        if t=='A': return 'Skinny'
        if t=='B': return 'Petite'
        if t=='C' and shape != 'Hourglass': return 'Average'
        if t=='C' and shape == 'Hourglass': return 'Curvy'
        if t=='D' and shape == 'Banana': return 'BBW'
        if t=='D' and shape == 'Hourglass': return 'BBW - Curvy'
        if t=='D' and shape == 'Pear': return 'BBW - Bottom Heavy'
        if t=='D' and shape == 'Apple': return 'BBW - Top Heavy'
        if t=='E' and (shape in ['Banana','Hourglass']): return 'SSBBW'
        if t=='E' and shape == 'Apple': return 'SSBBW - Top Heavy'
        if t=='E' and shape == 'Pear': return 'SSBBW - Bottom Heavy'
        return 'Average'
    except Exception:
        return 'Error!'

# ------------------------------------------------------------------------------
# Auth routes (guarded if db is None)
# ------------------------------------------------------------------------------
@api_router.post("/auth/signup", response_model=dict)
async def signup(user_data: UserCreate):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user_data.password)
    user = User(email=user_data.email, hashed_password=hashed_password)
    await db.users.insert_one(user.dict())
    return {"message": "User created successfully"}

@api_router.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    user = await db.users.find_one({"email": user_data.email})
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["email"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return {"email": current_user.email, "id": current_user.id}

# ------------------------------------------------------------------------------
# Skin tone & outfits
# ------------------------------------------------------------------------------
@api_router.post("/analyze-skin-tone")
async def analyze_skin_tone(
    file: UploadFile = File(...),
    current_user: Optional[User] = Depends(get_current_user)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    detected_color, recommended_colors = detect_skin_tone(image_bytes)
    if db:
        analysis = SkinToneAnalysis(
            user_id=current_user.id if current_user else None,
            detected_skin_tone=detected_color,
            recommended_colors=recommended_colors
        )
        await db.skin_tone_analyses.insert_one(analysis.dict())
        analysis_id = analysis.id
    else:
        analysis_id = None
    return {
        "detected_skin_tone": detected_color,
        "recommended_colors": recommended_colors,
        "analysis_id": analysis_id
    }

@api_router.get("/outfit-recommendations")
async def get_outfit_recommendations(
    gender: str,
    recommended_colors: str,
    limit: int = 5
):
    if styles_df.empty:
        raise HTTPException(status_code=500, detail="Fashion dataset not available")
    colors_list = [c.strip() for c in recommended_colors.split(',')]
    filtered_df = styles_df[
        (styles_df['gender'].str.lower() == gender.lower()) &
        (styles_df['baseColour'].isin(colors_list))
    ]
    if filtered_df.empty:
        filtered_df = styles_df[styles_df['gender'].str.lower() == gender.lower()]
    sample_size = min(limit, len(filtered_df))
    sampled_items = filtered_df.sample(n=sample_size) if sample_size > 0 else filtered_df
    recommendations = []
    for _, item in sampled_items.iterrows():
        rec = OutfitRecommendation(
            item_id=str(item['id']),
            product_name=item['productDisplayName'],
            category=item['masterCategory'],
            sub_category=item['subCategory'],
            article_type=item['articleType'],
            base_colour=item['baseColour'],
            gender=item['gender'],
            season=item['season'],
            usage=item['usage']
        ).dict()
        rec['image_url'] = get_item_image(item['articleType'])
        recommendations.append(rec)
    return {"recommendations": recommendations}

# ------------------------------------------------------------------------------
# Body Type API (accepts JSON or form/multipart)
# ------------------------------------------------------------------------------
@api_router.post("/bodytype", response_model=BodyTypeResponse)
async def bodytype_api(
    request: Request,
    height: Optional[str] = Form(None),
    weight: Optional[str] = Form(None),
    bust:   Optional[str] = Form(None),
    cup:    Optional[str] = Form(None),
    waist:  Optional[str] = Form(None),
    hip:    Optional[str] = Form(None),
):
    """Accepts either JSON or form/multipart with fields:
    height, weight, bust, cup, waist, hip
    """
    # If any form values are present, prefer them
    if any(v is not None for v in [height, weight, bust, cup, waist, hip]):
        payload = BodyTypeRequest(
            height=height or "",
            weight=weight or "",
            bust=bust or "",
            cup=cup or "",
            waist=waist or "",
            hip=hip or "",
        )
    else:
        # Otherwise try JSON
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid body. Send JSON or form-data with fields: height, weight, bust, cup, waist, hip.",
            )
        try:
            payload = BodyTypeRequest(**data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Malformed JSON for bodytype: {e}")

    # Calculate
    bmi = _get_bmi(payload.weight, payload.height)
    mult = _get_breast_multiplier(payload.bust, payload.cup)
    breast_desc = _get_breast_desc(mult)
    butt_desc = _get_butt_desc(payload.hip)
    shape = _get_body_shape(payload.bust, payload.waist, payload.hip)
    btype = _get_body_type(bmi, shape)

    return BodyTypeResponse(
        bmi=bmi,
        breast_multiplier=mult,
        breast_desc=breast_desc,
        butt_desc=butt_desc,
        body_shape=shape,
        body_type=btype,
    )

# ------------------------------------------------------------------------------
# Favorites (only if db configured)
# ------------------------------------------------------------------------------
@api_router.post("/favorites")
async def add_to_favorites(
    item_id: str,
    product_name: str,
    base_colour: str,
    current_user: User = Depends(get_current_user)
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    existing = await db.favorites.find_one({
        "user_id": current_user.id,
        "item_id": item_id
    })
    if existing:
        raise HTTPException(status_code=400, detail="Item already in favorites")
    favorite = Favorite(
        user_id=current_user.id,
        item_id=item_id,
        product_name=product_name,
        base_colour=base_colour
    )
    await db.favorites.insert_one(favorite.dict())
    return {"message": "Added to favorites"}

@api_router.get("/favorites")
async def get_favorites(current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    favorites = await db.favorites.find({"user_id": current_user.id}).to_list(100)
    return {"favorites": favorites}

@api_router.delete("/favorites/{item_id}")
async def remove_from_favorites(
    item_id: str,
    current_user: User = Depends(get_current_user)
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    result = await db.favorites.delete_one({
        "user_id": current_user.id,
        "item_id": item_id
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Favorite not found")
    return {"message": "Removed from favorites"}

# ------------------------------------------------------------------------------
# General routes
# ------------------------------------------------------------------------------
@api_router.get("/")
async def root():
    return {"message": "Fashion Recommendation API"}

@api_router.get("/fashion-categories")
async def get_fashion_categories():
    if styles_df.empty:
        return {"categories": []}
    categories = styles_df.groupby(['masterCategory', 'subCategory']).size().reset_index()
    category_data = []
    for _, row in categories.iterrows():
        category_data.append({
            "master_category": row['masterCategory'],
            "sub_category": row['subCategory'],
            "count": row[0]
        })
    return {"categories": category_data}

# ------------------------------------------------------------------------------
# Mount router & middleware
# ------------------------------------------------------------------------------
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Shutdown
# ------------------------------------------------------------------------------
@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        client.close()