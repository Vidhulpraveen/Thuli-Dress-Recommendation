import os
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

def generate_description_for_image(image_path):
    # Initialize Vertex AI
    vertexai.init(project="test-hack-429406", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-002")

    # Open the image file and convert it to base64
    base64_image_data = image_to_base64(image_path)

    # Create the Part with decoded image data
    image_part = Part.from_data(
        mime_type="image/jpeg",
        data=base64.b64decode(base64_image_data),
    )

    # Define prompt for the model
    prompt = """Describe the outfit worn by the individual also describe the gender, body type and facial features. 
    Body shape may be of classes: Athletic, Hourglass, Apple, Banana, Inverted Triangle, Rectangle, Trapezoid etc... 
    Facial features contain face shape, jawline type, eye color etc... Face shape may be of classes: oval, round, square etc... 
    Jawline type may be of classes: defined, sharp, soft. Eye color may be brown, blue, black, green etc.. Hair type may be of 
    classes: wavy, straight, curly etc.. also give me the occasion in which the outfit is worn. Don't have bold words in the output; 
    just have the description as a paragraph."""

    # Generate content using the model
    responses = model.generate_content(
        [prompt, image_part],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,  # stream=False for processing all at once
    )

    generated_text = responses.candidates[0].content.parts[0].text
    return (generated_text)

def image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

def process_images_in_directory(input_dir, output_dir):
    """Recursively process all images in the input directory, generate descriptions, and save them as text files."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image file
                image_path = os.path.join(root, file)
                
                # Generate a description for the image
                description = generate_description_for_image(image_path)

                # Create the output directory mirroring the input directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Create the output text file with the same name as the image
                output_file_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.txt")
                with open(output_file_path, "w") as f:
                    f.write(description)

                print(f"Processed: {image_path} -> {output_file_path}")

# Configuration for the Gemini model
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# Safety settings for content generation
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

# Example usage
if __name__ == "__main__":
    input_directory = "./vector_dataset/images"  # Path to the folder containing images
    output_directory = "./vector_dataset/description"     # Path to the folder to save descriptions

    # Process images in the input directory and save descriptions in the output directory
    process_images_in_directory(input_directory, output_directory)
