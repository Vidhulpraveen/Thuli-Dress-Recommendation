import requests
import base64
from PIL import Image
from io import BytesIO
import io

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


def getImage(text_description):
    input_prompt = "An individual wearing an attire with the following description: " + text_description
    image_bytes = query({
        "inputs": input_prompt,
    })

    image = Image.open(io.BytesIO(image_bytes))
    output_file = "processed-images/generated_outfit.png"
    image.save(output_file)
    return output_file