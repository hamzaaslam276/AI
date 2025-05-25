import cv2
import pytesseract
import easyocr
import numpy as np
import pandas as pd
import openai
from PIL import Image
import re
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

reader = easyocr.Reader(['en'])

def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text(image):
    result = reader.readtext(image, detail=0)
    return " ".join(result)

def structure_data(text):
    # Example regex for extracting test results
    pattern = r"([A-Za-z ]+)[^\d]*(\d+\.?\d*)[^\d]*(\d+\.?\d*)-(\d+\.?\d*)[^\w]*(mg/dL|g/dL|%)?"
    matches = re.findall(pattern, text)
    data = []
    for match in matches:
        data.append({
            "Test Name": match[0].strip(),
            "Measured Value": float(match[1]),
            "Normal Low": float(match[2]),
            "Normal High": float(match[3]),
            "Unit": match[4]
        })
    return pd.DataFrame(data)

def generate_explanations(df):
    explanations = []
    for _, row in df.iterrows():
        prompt = (
            f"Explain in simple terms what it means if the patient's {row['Test Name']} is "
            f"{row['Measured Value']} {row['Unit']} given the normal range is "
            f"{row['Normal Low']}–{row['Normal High']} {row['Unit']}."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            explanation = response.choices[0].message.content.strip()
        except Exception as e:
            explanation = f"Error generating explanation: {e}"
        explanations.append({
            "Test Name": row['Test Name'],
            "Explanation": explanation
        })
    return explanations
