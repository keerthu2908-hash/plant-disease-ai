"""
image_predictor.py — Strict, confidence-aware, multi-crop image prediction module.
 
Architecture:
  1. Load a general plant disease classifier (multi-crop, not rice-only)
  2. Normalize predicted labels to match dataset naming conventions
  3. Apply strict confidence thresholds before trusting any prediction
  4. Check crop relevance — reject labels that don't match the selected crop
  5. Return structured results with confidence tiers and trust flags
 
Confidence tiers:
  >= 0.90  →  STRONG   — image evidence is highly trustworthy
  0.75–0.89 → MODERATE — supporting evidence only, do not override symptoms
  < 0.75   →  WEAK     — do not trust, fall back to symptom retrieval
"""
 
from PIL import Image
from transformers import pipeline
import re
 
# ─────────────────────────────────────────────
# CONFIDENCE THRESHOLDS (strict)
# ─────────────────────────────────────────────
THRESHOLD_STRONG = 0.90
THRESHOLD_MODERATE = 0.75
THRESHOLD_WEAK = 0.75  # below this = do not trust
 
# ─────────────────────────────────────────────
# MODEL LOADING (cached at module level)
# ─────────────────────────────────────────────
# Using a general multi-crop plant disease model instead of rice-only.
# Replace this model ID with your own fine-tuned model when available.
_classifier = None
 
def _get_classifier():
    """Lazy-load the image classifier so import doesn't block."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "image-classification",
            model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
            top_k=5,
        )
    return _classifier
 
 
# ─────────────────────────────────────────────
# LABEL NORMALIZATION
# ─────────────────────────────────────────────
# Maps raw model output labels → canonical names that match your dataset.
# Add more aliases as you discover mismatches.
LABEL_ALIASES = {
    # Tomato
    "tomato early blight":        "Early Blight",
    "tomato late blight":         "Late Blight",
    "tomato leaf mold":           "Leaf Mould",
    "tomato septoria leaf spot":  "Septoria Leaf Spot",
    "tomato bacterial spot":      "Bacterial Spot",
    "tomato yellow leaf curl":    "Yellow Leaf Curl Virus",
    "tomato mosaic virus":        "Mosaic Virus",
    "tomato target spot":         "Target Spot",
    "tomato spider mites":        "Spider Mites",
    # Rice
    "rice blast":                 "Blast",
    "rice brown spot":            "Brown Spot",
    "rice leaf blight":           "Bacterial Leaf Blight",
    "rice sheath blight":         "Sheath Blight",
    "rice hispa":                 "Spiny beetle / Hispa",
    "rice tungro":                "Tungro",
    # Potato
    "potato early blight":        "Early Blight",
    "potato late blight":         "Late Blight",
    # Corn / Maize
    "corn common rust":           "Common Rust",
    "corn northern leaf blight":  "Northern Leaf Blight",
    "corn cercospora leaf spot":  "Cercospora Leaf Spot",
    "corn gray leaf spot":        "Grey Leaf Spot",
    "maize common rust":          "Common Rust",
    "maize northern leaf blight": "Northern Leaf Blight",
    # Cotton
    "cotton aphid":               "Cotton aphid",
    "cotton bollworm":            "American boll worm",
    # Apple
    "apple scab":                 "Apple Scab",
    "apple black rot":            "Black Rot",
    "apple cedar rust":           "Cedar Apple Rust",
    # Grape
    "grape black rot":            "Black Rot",
    "grape esca":                 "Esca (Black Measles)",
    "grape leaf blight":          "Leaf Blight",
    # General
    "healthy":                    "Healthy",
    "background":                 "Background",
}
 
# Crop keywords extracted from model labels (model often prefixes crop name)
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
 
 
def normalize_label(raw_label: str) -> str:
    """
    Clean a raw model label into a human-readable, dataset-aligned name.
    Steps:
      1. Replace underscores/hyphens with spaces
      2. Collapse whitespace
      3. Lowercase for alias lookup
      4. Return alias if found, else title-case the cleaned label
    """
    cleaned = raw_label.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    lookup_key = cleaned.lower()
 
    if lookup_key in LABEL_ALIASES:
        return LABEL_ALIASES[lookup_key]
 
    # Try partial match (e.g., "Tomato___Early_Blight" → "tomato early blight")
    for alias_key, alias_val in LABEL_ALIASES.items():
        if alias_key in lookup_key or lookup_key in alias_key:
            return alias_val
 
    return cleaned.title()
 
 
def extract_crop_from_label(raw_label: str) -> str:
    """
    Try to extract a crop name from the model's predicted label.
    E.g., "Tomato___Early_Blight" → "Tomato"
    Returns empty string if no crop detected.
    """
    label_lower = raw_label.lower().replace("_", " ").replace("-", " ")
    for crop_key, crop_names in CROP_KEYWORDS_IN_LABELS.items():
        for name in crop_names:
            if name.lower() in label_lower:
                return name
    return ""
 
 
def check_crop_relevance(predicted_label: str, selected_crop: str) -> bool:
    """
    Check if the predicted label is relevant to the user's selected crop.
    Returns True if:
      - selected_crop is "All" (no filter)
      - the label contains the crop name
      - the label's extracted crop matches selected_crop
    """
    if not selected_crop or selected_crop.lower() == "all":
        return True
 
    label_lower = predicted_label.lower().replace("_", " ").replace("-", " ")
    crop_lower = selected_crop.lower()
 
    # Direct crop name in label
    if crop_lower in label_lower:
        return True
 
    # Check via crop keywords
    for crop_key, crop_names in CROP_KEYWORDS_IN_LABELS.items():
        if crop_lower in [n.lower() for n in crop_names] or crop_lower == crop_key:
            for name in crop_names:
                if name.lower() in label_lower:
                    return True
 
    # If label doesn't contain ANY known crop name, it might be generic — allow it
    has_any_crop = any(
        name.lower() in label_lower
        for names in CROP_KEYWORDS_IN_LABELS.values()
        for name in names
    )
    if not has_any_crop:
        return True  # generic label like "Leaf Blight" — could apply to any crop
 
    return False
 
 
def get_confidence_tier(score: float) -> str:
    """Return confidence tier string."""
    if score >= THRESHOLD_STRONG:
        return "STRONG"
    elif score >= THRESHOLD_MODERATE:
        return "MODERATE"
    return "WEAK"
 
 
def label_matches_dataset(normalized_label: str, dataset_records: list) -> bool:
    """
    Check if the normalized label has meaningful overlap with any record
    in the dataset (name, scientific_name, cause_description).
    Uses token overlap — at least 1 significant word must match.
    """
    if not dataset_records:
        return False
 
    label_tokens = set(normalized_label.lower().split())
    # Remove very common stop words
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
 
 
# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_disease_from_image(
    uploaded_file,
    selected_crop: str = "All",
    dataset_records: list = None,
) -> dict:
    """
    Strict, confidence-aware image prediction pipeline.
 
    Returns a dict with:
      - predictions: list of {label, normalized_label, score, confidence_tier, crop_relevant, dataset_match}
      - best_prediction: the single best trusted prediction (or None)
      - trust_image: bool — should the app trust image evidence?
      - trust_level: "STRONG" / "MODERATE" / "WEAK" / "NONE"
      - warning: str — human-readable warning if confidence is low
    """
    result = {
        "predictions": [],
        "best_prediction": None,
        "trust_image": False,
        "trust_level": "NONE",
        "warning": "No image prediction available.",
    }
 
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        result["warning"] = f"Failed to read image: {e}"
        return result
 
    try:
        classifier = _get_classifier()
        raw_predictions = classifier(image)
    except Exception as e:
        result["warning"] = f"Image model failed: {e}"
        return result
 
    if not raw_predictions:
        result["warning"] = "Image model returned no predictions."
        return result
 
    dataset_records = dataset_records or []
 
    # ── Process each prediction ──
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
 
    # ── Select best trusted prediction ──
    # Priority: STRONG + crop_relevant + dataset_match > STRONG + crop_relevant > MODERATE + crop_relevant
    best = None
    for p in processed:
        if p["normalized_label"].lower() in ("healthy", "background"):
            continue  # skip non-disease labels for best prediction
 
        if p["confidence_tier"] == "STRONG" and p["crop_relevant"]:
            best = p
            break
        elif p["confidence_tier"] == "MODERATE" and p["crop_relevant"] and best is None:
            best = p
 
    if best:
        result["best_prediction"] = best
        if best["confidence_tier"] == "STRONG":
            result["trust_image"] = True
            result["trust_level"] = "STRONG"
            result["warning"] = ""
        elif best["confidence_tier"] == "MODERATE":
            result["trust_image"] = False  # don't override symptoms
            result["trust_level"] = "MODERATE"
            result["warning"] = (
                "Image confidence is moderate. Diagnosis relies more on symptom matching. "
                "Image prediction is used as supporting evidence only."
            )
    else:
        # Check if top prediction is at least present
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
    """
    Returns a weight (0.0–1.0) for how much the image prediction should
    influence the final combined query / ranking.
 
    Fusion policy:
      STRONG  + symptoms → 0.50 (equal weight)
      STRONG  + no symptoms → 0.80 (lean on image)
      MODERATE + symptoms → 0.25 (symptoms dominate)
      MODERATE + no symptoms → 0.55
      WEAK    + symptoms → 0.10 (nearly ignore image)
      WEAK    + no symptoms → 0.35 (cautious)
      NONE    → 0.00
    """
    weights = {
        "STRONG":   (0.50, 0.80),
        "MODERATE": (0.25, 0.55),
        "WEAK":     (0.10, 0.35),
        "NONE":     (0.00, 0.00),
    }
    with_sym, without_sym = weights.get(trust_level, (0.00, 0.00))
    return with_sym if has_symptoms else without_sym