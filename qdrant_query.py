import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np


def generate(project_id, location):
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

# Open an image file and convert it to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

# Function to query Qdrant for relevant outfit descriptions
def search_outfit_descriptions(individual_description, query_input, model, client, collection_name, top_k=5):
    # Convert the user's query to an embedding
    query = individual_description + query_input
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize the embedding

    # Search Qdrant for the top_k most similar descriptions
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    # Retrieve and return the descriptions and similarity scores
    retrieved_descriptions = [(result.payload["description"], result.score) for result in search_results]
    
    return retrieved_descriptions

def generate_outfit_suggestion_with_gemini(retrieved_descriptions, individual_description, user_query, project_id, location, model_id="gemini-1.5-flash-002"):
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Load the Generative Model
    model = GenerativeModel(model_id)

    # Prepare the input prompt
    descriptions_context = "\n".join([desc for desc, _ in retrieved_descriptions])
    
    # Full prompt including individual and outfit descriptions
    prompt = (
        f"Based on the following outfit descriptions:\n{descriptions_context} and\n"
        f"Based on the following individual descriptions:\n{individual_description}\n"
        f"Suggest a personalized outfit for the following request: {user_query}."
    )

    # Construct generation configuration
    generation_config = {
        "max_output_tokens": 200,  # Adjust token limits
        "temperature": 0.7         # Adjust randomness of the model
    }

    # Generate content using the Gemini model
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    for response in responses:
        print(response.text, end="")


# Initialize Qdrant Cloud client (replace 'your-cluster-url' and 'your-api-key')
client = QdrantClient(
    url="https://cc595f43-0fbd-4cb0-9dce-b53a64920c4d.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="5d-hyrAIstlGzaVnXNeRJH4LXct9i26SFnrvDPZfu1VMb2pssYqHSQ",
)

# Initialize the Sentence Transformer model for generating text embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
collection_name = "outfit_descriptions"
# Example query from the user

# image_input = input("Image Name: ")
# query_input = input("Outfit Description: ")


image_input = "20PC39.heic"
query_input = "Casual outing"

#user_query = "suggest an outfit for a formal wedding"

base64_image_data = image_to_base64(image_input)

# Create the Part with decoded image data
image_part = Part.from_data(
    mime_type="image/jpeg",
    data=base64.b64decode(base64_image_data),
)

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

project_id="test-hack-429406"
location="us-central1"

individual_description = generate(project_id, location)

# Example of retrieved descriptions from Qdrant Cloud
retrieved_descriptions = search_outfit_descriptions(individual_description, query_input, model, client, collection_name)

# Generate the outfit suggestion
suggestion = generate_outfit_suggestion_with_gemini(retrieved_descriptions, individual_description, query_input, project_id, location)
print("Outfit Suggestion:", suggestion)