# Imports
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np


app = FastAPI()

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():    
    with open(Path(__file__).parent / "static/index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    
    return HTMLResponse(content=html_content, headers={"Content-Type": "text/html; charset=utf-8"})


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), filter: str = Form(...)):
    image = Image.open(file.file)

    if filter == "blur":
        image = apply_blur(image)
    elif filter == "sharp":
        image = apply_sharp(image)
    elif filter == "rotate":
        image = apply_rotation(image)

    img_byte_arr = image_to_bytes(image)
    return StreamingResponse(img_byte_arr, media_type="image/png")


# Funções Auxiliares
def apply_blur(image: Image.Image) -> Image.Image:
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    blurred_image = cv2.GaussianBlur(image_cv, (15, 15), 0)  
    blurred_image_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    
    return blurred_image_pil


def apply_sharp(image: Image.Image) -> Image.Image:
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

    imagem_sharpened = cv2.filter2D(image_cv, -1, kernel)
    sharp_image_pil = Image.fromarray(cv2.cvtColor(imagem_sharpened, cv2.COLOR_BGR2RGB))
    
    return sharp_image_pil


def apply_rotation(image: Image.Image) -> Image.Image:
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  
    
    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, 90, 1.0)  
    rotated_image = cv2.warpAffine(image_cv, M, (w, h))  
    
    rotated_image_pil = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    
    return rotated_image_pil


def image_to_bytes(image: Image.Image) -> BytesIO:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return img_byte_arr
