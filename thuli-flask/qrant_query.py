import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np
import os
import re

project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
qrant_api_key = os.getenv("QDRANT_API_KEY")
qrant_url = os.getenv("QRANT_URL")
client = QdrantClient(
    url = qrant_url, 
    api_key = qrant_api_key,
)

collection_name = "outfit_descriptions"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]


def process_output(text):
    options = re.findall(r'Option \d+:(.*?)(?=Option \d+:|$)', text, re.DOTALL)

    options = [option.strip() for option in options]

    return options


def generate(image_part):
    vertexai.init(project=project_id, location=location)

    model = GenerativeModel(
        "gemini-1.5-flash-002",
    )
    responses = model.generate_content(
        ["""Describe the the individual describe the gender, body type and facial features. Body shape may be of
        classes: Athletic, Hourglass, Apple, Banana, Inverted Triangle, Rectangle, Trapezoid etc... Facial features contain face shape, jawline type,
        eye color etc... Face shape may be of classes: oval, round, square etc... Jawline type may be of classes: defined, sharp, soft. Eye color may
        be brown, blue, black, green etc.. Hair type may be of classes: wavy, straight, curly etc. Don't have bold words in the output just have the description as paragraph""", image_part],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    answer = ""
    for response in responses:
        answer += response.text

    return answer

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

def search_outfit_descriptions(individual_description, query_input, top_k=5):
    query = individual_description + query_input
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding) 

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    retrieved_descriptions = [(result.payload["description"], result.score) for result in search_results]
    
    return retrieved_descriptions

def generate_outfit_suggestion_with_gemini(retrieved_descriptions, individual_description, user_query, model_id="gemini-1.5-flash-002"):
    vertexai.init(project=project_id, location=location)

    model = GenerativeModel(model_id)

    descriptions_context = "\n".join([desc for desc, _ in retrieved_descriptions])
    
    prompt = (
        f"Based on the following outfit descriptions:\n{descriptions_context} and\n"
        f"Based on the following individual descriptions:\n{individual_description}\n"
        f"Suggest a personalized outfit for the following request: {user_query}.\n"
        f"Suggest options such as Option 1, Option 2. Limit to 2 options."
        f"Avoid bold words in the output, the heading should be the option and the type of attire followed by a paragraph explaining the attire."
    )

    generation_config = {
        "max_output_tokens": 200,
        "temperature": 0.7     
    }

    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    final_result = ''
    for response in responses:
        final_result += response.text

    return final_result


def get_suggestions(image_path, query_input):
    base64_image_data = image_to_base64(image_path)
    image_part = Part.from_data(
        mime_type="image/jpeg",
        data=base64.b64decode(base64_image_data),
    )
    individual_description = generate(image_part)
    retrieved_descriptions = search_outfit_descriptions(individual_description, query_input)
    suggestion = generate_outfit_suggestion_with_gemini(retrieved_descriptions, individual_description, query_input)
    final_output = process_output(suggestion)
    return final_output