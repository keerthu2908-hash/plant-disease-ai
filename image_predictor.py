from transformers import pipeline
from PIL import Image

classifier = pipeline(
    "image-classification",
    model="haiderAI/vision-transformer-rice-disease-detection"
)

def predict_disease_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return classifier(image)[:3]