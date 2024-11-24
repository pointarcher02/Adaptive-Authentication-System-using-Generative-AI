


from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.model import VAE, check_anomaly
import google.generativeai as genai
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

import logging

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/authenticate", response_class=HTMLResponse)
async def get_authenticate_form():
    with open("static/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

logging.basicConfig(level=logging.INFO)


# Set up CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the trained VAE model and scaler parameters
vae = VAE.load_model("scripts/model.pth")
scaler_mean = np.load("data/scaler_params.npy")
scaler_scale = np.load("data/scaler_scale.npy")
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Configure the Google Generative AI client
genai.configure(api_key="AIzaSyCC7URiovOJEeHCVUyQLqMEk6LusDo3n-E")
model = genai.GenerativeModel("gemini-1.5-flash")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/authenticate")
async def authenticate(request: Request):
    try:
        data = await request.json()
        if 'location' not in data:
            return JSONResponse(status_code=400, content={"error": "Location data is missing"})

        anomaly_score = check_anomaly(data, vae)
        logging.info(f"Anomaly Score: {anomaly_score}")

        threshold = 1500  # Define a threshold to detect anomaly
        if anomaly_score < threshold:
            return {"access_granted": True}
        else:
            explanation = generate_explanation(data, anomaly_score)
            next_step = get_next_verification_step(data, anomaly_score)
            return {
                "access_granted": False,
                "additional_authentication": True,
                "explanation": explanation,
                "next_step": next_step
            }
    except Exception as e:
        logging.error("Failed to process authentication:", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# Function to generate natural language explanation
def generate_explanation(data, anomaly_score):
    prompt = f"""
    An anomaly was detected based on the following user data:
    Typing Speed: {data['typing_speed']}
    Mouse Movement: {data['mouse_movement']}
    Location: {data['location']}
    Anomaly Score: {anomaly_score:.2f}

    Please provide a user-friendly explanation for why this behavior might be considered unusual.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "No explanation available."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# Function to decide the next verification step
def get_next_verification_step(data, anomaly_score):
    prompt = f"""
    An anomaly has been detected for the user with the following details:
    - Typing Speed: {data['typing_speed']}
    - Mouse Movement: {data['mouse_movement']}
    - Location: {data['location']}
    - Anomaly Score: {anomaly_score:.2f}

    Considering user context, provide the best next authentication step. Options include:
    1. Send OTP to registered mobile.
    2. Ask a security question.
    3. Email verification link.
    4. Require biometric authentication.

    Provide the best choice with a brief reason.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "No next step recommendation available."
    except Exception as e:
        return f"Error generating next verification step: {str(e)}"
