"""
image_predictor.py — Strict, confidence-aware, multi-crop image prediction module.
"""

from PIL import Image
from transformers import AutoModelForImageClassification
import torch
from torchvision import transforms
import re

THRESHOLD_STRONG = 0.90
THRESHOLD_MODERATE = 0.75
THRESHOLD_WEAK = 0.75

_model_mobilenet = None
_model_resnet = None
_model_efficientnet = None
_processor = None


class ManualImageProcessor:
    """
    Minimal callable processor for models that do not ship a valid HF image processor config.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, images, return_tensors="pt"):
        if isinstance(images, Image.Image):
            tensor = self.transform(images).unsqueeze(0)
        else:
            raise ValueError("ManualImageProcessor expects a PIL image.")
        return {"pixel_values": tensor}


def _get_classifiers():
    global _model_mobilenet, _model_resnet, _model_efficientnet, _processor

    if _model_mobilenet is None:
        _model_mobilenet = AutoModelForImageClassification.from_pretrained(
            "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        )
        _model_mobilenet.eval()

    if _model_resnet is None:
        _model_resnet = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50"
        )
        _model_resnet.eval()

    if _model_efficientnet is None:
        _model_efficientnet = AutoModelForImageClassification.from_pretrained(
            "google/efficientnet-b0"
        )
        _model_efficientnet.eval()

    if _processor is None:
        _processor = ManualImageProcessor()

    return _model_mobilenet, _model_resnet, _model_efficientnet, _processor


def _get_classifier():
    # Backward-compatible single-model accessor.
    model_mobilenet, _, _, processor = _get_classifiers()
    return model_mobilenet, processor


def get_loaded_model():
    return _get_classifier()


LABEL_ALIASES = {
    "tomato early blight": "Early Blight",
    "tomato late blight": "Late Blight",
    "tomato leaf mold": "Leaf Mould",
    "tomato septoria leaf spot": "Septoria Leaf Spot",
    "tomato bacterial spot": "Bacterial Spot",
    "tomato yellow leaf curl": "Yellow Leaf Curl Virus",
    "tomato mosaic virus": "Mosaic Virus",
    "tomato target spot": "Target Spot",
    "tomato spider mites": "Spider Mites",
    "rice blast": "Blast",
    "rice brown spot": "Brown Spot",
    "rice leaf blight": "Bacterial Leaf Blight",
    "rice sheath blight": "Sheath Blight",
    "rice hispa": "Spiny beetle / Hispa",
    "rice tungro": "Tungro",
    "potato early blight": "Early Blight",
    "potato late blight": "Late Blight",
    "corn common rust": "Common Rust",
    "corn northern leaf blight": "Northern Leaf Blight",
    "corn cercospora leaf spot": "Cercospora Leaf Spot",
    "corn gray leaf spot": "Grey Leaf Spot",
    "maize common rust": "Common Rust",
    "maize northern leaf blight": "Northern Leaf Blight",
    "cotton aphid": "Cotton aphid",
    "cotton bollworm": "American boll worm",
    "apple scab": "Apple Scab",
    "apple black rot": "Black Rot",
    "apple cedar rust": "Cedar Apple Rust",
    "grape black rot": "Black Rot",
    "grape esca": "Esca (Black Measles)",
    "grape leaf blight": "Leaf Blight",
    "healthy": "Healthy",
    "background": "Background",
}

CROP_KEYWORDS_IN_LABELS = {
    "tomato": ["Tomato"],
    "rice": ["Rice", "Paddy"],
    "potato": ["Potato"],
    "corn": ["Maize", "Corn"],
    "maize": ["Maize", "Corn"],
    "cotton": ["Cotton"],
    "apple": ["Apple"],
    "grape": ["Grape"],
    "soybean": ["Soybean", "Soy"],
    "groundnut": ["Groundnut", "Peanut"],
    "sugarcane": ["Sugarcane"],
    "wheat": ["Wheat"],
    "sorghum": ["Sorghum", "Jowar"],
}


NON_DISEASE_LABELS = {"healthy", "background"}


def normalize_label(raw_label: str) -> str:
    cleaned = raw_label.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    lookup_key = cleaned.lower()

    if lookup_key in LABEL_ALIASES:
        return LABEL_ALIASES[lookup_key]

    for alias_key, alias_val in LABEL_ALIASES.items():
        if alias_key in lookup_key or lookup_key in alias_key:
            return alias_val

    return cleaned.title()


def check_crop_relevance(predicted_label: str, selected_crop: str) -> bool:
    if not selected_crop or selected_crop.lower() == "all":
        return True

    label_lower = predicted_label.lower().replace("_", " ").replace("-", " ")
    crop_lower = selected_crop.lower()

    supported_crop_names = {
        name.lower()
        for names in CROP_KEYWORDS_IN_LABELS.values()
        for name in names
    }
    supported_crop_keys = {key.lower() for key in CROP_KEYWORDS_IN_LABELS}

    # If selected crop is supported, use normal matching.
    if crop_lower in supported_crop_keys or crop_lower in supported_crop_names:
        if crop_lower in label_lower:
            return True

        for crop_key, crop_names in CROP_KEYWORDS_IN_LABELS.items():
            if crop_lower in [n.lower() for n in crop_names] or crop_lower == crop_key:
                for name in crop_names:
                    if name.lower() in label_lower:
                        return True

        has_any_crop = any(
            name.lower() in label_lower
            for names in CROP_KEYWORDS_IN_LABELS.values()
            for name in names
        )
        if not has_any_crop:
            return True

        return False

    # If selected crop is unsupported (for example, okra), reject labels
    # that explicitly mention any known other crop.
    has_known_other_crop = any(
        name.lower() in label_lower
        for names in CROP_KEYWORDS_IN_LABELS.values()
        for name in names
    )

    if has_known_other_crop:
        return False

    return True


def get_first_usable_image_label(image_result: dict) -> str:
    """
    Return the first non-healthy, non-background normalized label
    from best_prediction or predictions list.
    """
    if not image_result:
        return ""

    best = image_result.get("best_prediction") or {}
    best_label = str(best.get("normalized_label") or best.get("label") or "").strip()
    if best_label and best_label.lower() not in NON_DISEASE_LABELS:
        return best_label

    for pred in image_result.get("predictions", []) or []:
        label = str(pred.get("normalized_label") or pred.get("label") or "").strip()
        if label and label.lower() not in NON_DISEASE_LABELS:
            return label

    return ""


def get_confidence_tier(score: float) -> str:
    if score >= THRESHOLD_STRONG:
        return "STRONG"
    elif score >= THRESHOLD_MODERATE:
        return "MODERATE"
    return "WEAK"


def label_matches_dataset(normalized_label: str, dataset_records: list) -> bool:
    if not dataset_records:
        return False

    label_tokens = set(normalized_label.lower().split())
    stop_words = {"the", "a", "an", "of", "in", "on", "and", "or", "is", "to", "for", "with"}
    label_tokens -= stop_words

    if not label_tokens:
        return False

    for record in dataset_records:
        for field in ["name", "scientific_name", "cause_description"]:
            field_val = str(record.get(field, "")).lower()
            field_tokens = set(field_val.split()) - stop_words
            if label_tokens & field_tokens:
                return True

    return False


def is_plant_image(image: Image.Image) -> bool:
    """
    Basic plant vs non-plant detection.
    Returns True if likely plant, else False.
    """
    try:
        img = image.resize((64, 64))
        pixels = list(img.getdata())

        green_pixels = 0
        for r, g, b in pixels:
            if g > r and g > b:
                green_pixels += 1

        ratio = green_pixels / len(pixels)
        return ratio > 0.15
    except Exception:
        return True


def predict_disease_from_image(uploaded_file, selected_crop: str = "All", dataset_records: list = None) -> dict:
    result = {
        "predictions": [],
        "best_prediction": None,
        "trust_image": False,
        "trust_level": "NONE",
        "warning": "No image prediction available.",
    }

    try:
        image = Image.open(uploaded_file).convert("RGB")

        if not is_plant_image(image):
            result["warning"] = "❌ Uploaded image does not appear to be a plant. Please upload a clear crop/leaf image."
            result["trust_level"] = "NONE"
            result["trust_image"] = False
            return result
    except Exception as e:
        result["warning"] = f"Failed to read image: {e}"
        return result

    try:
        model1, model2, model3, processor = _get_classifiers()
        inputs = processor(images=image, return_tensors="pt")

        def run_model(model, model_inputs):
            with torch.no_grad():
                outputs = model(**model_inputs)
            probs_tensor = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
            return probs_tensor

        valid_probs = []
        valid_models = []

        for model in [model1, model2, model3]:
            try:
                probs_tensor = run_model(model, inputs)

                if valid_probs and probs_tensor.shape != valid_probs[0].shape:
                    continue

                valid_probs.append(probs_tensor)
                valid_models.append(model)
            except Exception:
                continue

        if not valid_probs:
            result["warning"] = "Image model failed: no valid model outputs available."
            return result

        probs = sum(valid_probs) / len(valid_probs)

        top_k = min(5, probs.shape[0])
        top_scores, top_indices = torch.topk(probs, k=top_k)

        label_model = valid_models[0]
        id2label = getattr(label_model.config, "id2label", {}) or {}
        raw_predictions = [
            {
                "label": id2label.get(int(idx.item()), str(int(idx.item()))),
                "score": float(score.item()),
            }
            for score, idx in zip(top_scores, top_indices)
        ]
    except Exception as e:
        result["warning"] = f"Image model failed: {e}"
        return result

    if not raw_predictions:
        result["warning"] = "Image model returned no predictions."
        return result

    dataset_records = dataset_records or []
    processed = []

    for pred in raw_predictions[:5]:
        raw_label = pred.get("label", "")
        score = float(pred.get("score", 0.0))
        normalized = normalize_label(raw_label)
        tier = get_confidence_tier(score)
        crop_relevant = check_crop_relevance(raw_label, selected_crop)
        ds_match = label_matches_dataset(normalized, dataset_records)

        processed.append({
            "label": raw_label,
            "normalized_label": normalized,
            "score": score,
            "confidence_tier": tier,
            "crop_relevant": crop_relevant,
            "dataset_match": ds_match,
        })

    result["predictions"] = processed

    best = None
    for p in processed:
        if p["normalized_label"].lower() in NON_DISEASE_LABELS:
            continue

        if p["confidence_tier"] == "STRONG" and p["crop_relevant"]:
            best = p
            break
        elif p["confidence_tier"] == "STRONG" and best is None:
            best = p

    if best is None:
        for p in processed:
            if p["normalized_label"].lower() in NON_DISEASE_LABELS:
                continue
            if p["confidence_tier"] == "MODERATE" and p["crop_relevant"]:
                best = p
                break
            elif p["confidence_tier"] == "MODERATE" and best is None:
                best = p

    if best:
        result["best_prediction"] = best
        if best["confidence_tier"] == "STRONG":
            result["trust_image"] = True
            result["trust_level"] = "STRONG"
            result["warning"] = ""
        elif best["confidence_tier"] == "MODERATE":
            result["trust_image"] = False
            result["trust_level"] = "MODERATE"
            result["warning"] = (
                "Image confidence is moderate. Diagnosis relies more on symptom matching. "
                "Image prediction is used as supporting evidence only."
            )
    else:
        top = processed[0] if processed else None
        if top and top["score"] >= 0.50:
            result["best_prediction"] = top
            result["trust_level"] = "WEAK"
            result["trust_image"] = False
            result["warning"] = (
                "Image prediction confidence is low. The model is uncertain. "
                "Final diagnosis is based primarily on symptom text and crop filters."
            )
        else:
            result["trust_level"] = "NONE"
            result["trust_image"] = False
            result["warning"] = (
                "Image prediction is very uncertain. "
                "Diagnosis relies entirely on symptom text and dataset retrieval."
            )

    return result


def compute_image_weight(trust_level: str, has_symptoms: bool) -> float:
    weights = {
        "STRONG": (0.50, 0.80),
        "MODERATE": (0.25, 0.55),
        "WEAK": (0.10, 0.35),
        "NONE": (0.00, 0.00),
    }
    with_sym, without_sym = weights.get(trust_level, (0.00, 0.00))
    return with_sym if has_symptoms else without_sym
