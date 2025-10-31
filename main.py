from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()

# === スマホアプリ（React Native）からの通信を許可 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === フィルタ関数群 ===
def Deuteranope(im):
    b, g, r = cv2.split(im)
    bl = np.power((b/255+0.055)/1.055, 2.4)
    gl = np.power((g/255+0.055)/1.055, 2.4)
    rl = np.power((r/255+0.055)/1.055, 2.4)
    l = 0.31394*rl + 0.63957*gl + 0.04652*bl
    m = 0.15530*rl + 0.75796*gl + 0.08673*bl
    s = 0.01772*rl + 0.10945*gl + 0.87277*bl
    m = np.where(s <= l, 0.82781*l+0.17216*s, 0.81951*l+0.18046*s)
    rl = 5.47213*l - 4.64189*m + 0.16958*s
    gl = -1.12464*l + 2.29255*m - 0.16786*s
    bl = 0.02993*l - 0.19325*m + 1.16339*s
    bd = (np.power(bl,1/2.4)*1.055-0.055)*255
    gd = (np.power(gl,1/2.4)*1.055-0.055)*255
    rd = (np.power(rl,1/2.4)*1.055-0.055)*255
    return cv2.merge([
        bd.clip(0,255).astype(np.uint8),
        gd.clip(0,255).astype(np.uint8),
        rd.clip(0,255).astype(np.uint8)
    ])

def dark(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, -128*(a-133)/(164-133), 0).clip(-128, 0)
    l = (l.astype(np.float64) + x).clip(0, 255).astype(np.uint8)
    img = cv2.merge((l, a.astype(np.uint8), b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

def blue(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, -31*(a-133)/(164-133), 0).clip(-31, 0)
    b = (b.astype(np.float64) + x).clip(0, 255).astype(np.uint8)
    img = cv2.merge((l, a.astype(np.uint8), b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

def yellow(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, 31*(a-133)/(164-133), 0).clip(0, 31)
    b = (b.astype(np.float64) + x).clip(0, 255).astype(np.uint8)
    img = cv2.merge((l, a.astype(np.uint8), b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

FILTERS = {
    "Original": lambda x: x,
    "Deuteranope": Deuteranope,
    "Dark": dark,
    "Blue": blue,
    "Yellow": yellow,
}

# === フィルタAPI ===
@app.post("/filter/{filter_name}")
async def apply_filter(filter_name: str, file: UploadFile = File(...)):
    if filter_name not in FILTERS:
        return {"error": "Invalid filter name"}

    contents = await file.read()
    img = np.array(Image.open(BytesIO(contents)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    processed = FILTERS[filter_name](img)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    _, buffer = cv2.imencode(".jpg", processed)
    return {"image": buffer.tobytes().hex()}

@app.get("/")
def root():
    return {"message": "FastAPI 色覚補正サーバー 起動中 🚀"}
