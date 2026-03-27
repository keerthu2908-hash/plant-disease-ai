"""
merge_old_into_new.py
=====================
Merges an OLD disease JSON dataset into a NEW master disease JSON dataset.

WHAT IT DOES:
  1. Loads NEW_JSON (master_diseases.json) and OLD_JSON (disease_data.json)
  2. Maps OLD_JSON fields → NEW_JSON schema
  3. Skips non-disease entries (nutrient deficiencies, miscellaneous, etc.)
  4. Deduplicates using (crop + name + diagnosis_type + stage) — exact + fuzzy
  5. Adds only genuinely missing entries from OLD_JSON
  6. Fills management if missing using general scientifically correct practices
  7. Enforces strict key order on ALL entries
  8. Validates the final output and prints a full report

USAGE:
    python merge_old_into_new.py

INPUT FILES (same folder as this script):
    master_diseases.json   ← NEW_JSON  (your master dataset)
    disease_data.json      ← OLD_JSON  (legacy / supplementary dataset)

OUTPUT FILE (same folder):
    master_diseases.json   ← overwritten with merged result

KEY ORDER ENFORCED ON EVERY ENTRY:
    category, crop, diagnosis_type, stage, name,
    scientific_name, symptoms, management
"""

import json
import os
import re
import sys
from collections import Counter

# ── FILE PATHS ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_JSON_PATH = os.path.join(SCRIPT_DIR, "master_diseases.json")
OLD_JSON_PATH = os.path.join(SCRIPT_DIR, "disease_data.json")
OUTPUT_PATH   = os.path.join(SCRIPT_DIR, "master_diseases.json")  # overwrite in-place

# ── KEY ORDER ─────────────────────────────────────────────────────────────────
KEY_ORDER = [
    "category",
    "crop",
    "diagnosis_type",
    "stage",
    "name",
    "scientific_name",
    "symptoms",
    "management",
]

# ── VALID VALUES ──────────────────────────────────────────────────────────────
VALID_DIAGNOSIS_TYPES = {"Fungal", "Bacterial", "Viral", "Nematode", "Disease"}
VALID_STAGES          = {"Field", "PostHarvest"}

# ── DIAGNOSIS-TYPE MAP  (old 'type' → new 'diagnosis_type') ──────────────────
# None means "skip this entry" (not a biological disease)
TYPE_MAP = {
    "fungal"            : "Fungal",
    "fungal complex"    : "Fungal",
    "fungal (secondary)": "Fungal",
    "fungal/algal"      : "Fungal",
    "bacterial"         : "Bacterial",
    "phytoplasma"       : "Bacterial",   # phytoplasma → Bacterial (same convention)
    "viral"             : "Viral",
    "nematode"          : "Nematode",
    "parasitic"         : "Nematode",
    # ── skip these (not biological diseases) ──────────────────────────────────
    "nutrient"          : None,
    "deficiency"        : None,
    "abiotic"           : None,
}

# ── CATEGORY MAP  (old 'category' → new 'category') ──────────────────────────
# None means "skip this entry"
CATEGORY_MAP = {
    "cereal"        : "Cereal",
    "pulse"         : "Pulses",
    "cash crop"     : "Cash Crop",
    "oilseed"       : "Oilseed",
    "fruit"         : "Fruits",
    "flower"        : "Flower",
    "vegetable"     : "Vegetable",
    "plantation"    : "Plantation",
    "spice"         : "Spices",
    "spices"        : "Spices",
    "medicinal"     : "Medicinal",
    # ── skip these ────────────────────────────────────────────────────────────
    "miscellaneous" : None,
    "general"       : None,
}

# ── GENERAL MANAGEMENT FALLBACKS (used only when management is missing) ───────
GENERAL_MANAGEMENT = {
    "Fungal": [
        "Practice crop rotation with non-host crops to reduce soil-borne inoculum.",
        "Remove and destroy infected plant debris after harvest.",
        "Use region-specific or locally adapted resistant varieties where available.",
        "Ensure proper field drainage and avoid waterlogging.",
        "Apply a broad-spectrum fungicide such as Mancozeb 0.25% or Copper oxychloride 0.25% "
        "at recommended intervals on noticing initial symptoms.",
    ],
    "Bacterial": [
        "Use certified disease-free seed or planting material.",
        "Practice crop rotation with non-host crops.",
        "Remove and destroy infected plant parts promptly.",
        "Avoid overhead irrigation to reduce leaf wetness and splash spread.",
        "Apply copper-based bactericide or Streptocycline at recommended doses where appropriate.",
    ],
    "Viral": [
        "Use certified virus-free seed or planting material.",
        "Control insect vectors (aphids, whiteflies, thrips) with appropriate systemic insecticides.",
        "Remove and destroy infected plants promptly to prevent further spread.",
        "Maintain field sanitation and eliminate alternate weed hosts.",
        "Use region-specific or locally adapted resistant varieties where available.",
    ],
    "Nematode": [
        "Practice deep summer plowing to expose nematode eggs and juveniles to desiccation.",
        "Rotate with non-host crops or antagonistic crops such as marigold (Tagetes spp.).",
        "Apply neem cake to the soil at recommended rates.",
        "Use soil solarization for 4 to 6 weeks before planting in heavily infested fields.",
        "Apply recommended bio-control agents such as Purpureocillium lilacinum "
        "or Trichoderma spp. as soil treatment.",
    ],
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def fuzzy_normalize(s: str) -> str:
    """Strip punctuation/brackets on top of normalize — for fuzzy matching."""
    return normalize(re.sub(r"[\(\)\[\]/,\.\-]", " ", s))


def make_key(crop: str, name: str, dtype: str, stage: str = "field") -> tuple:
    return (normalize(crop), normalize(name), normalize(dtype), normalize(stage))


def make_fuzzy_key(crop: str, name: str, dtype: str, stage: str = "field") -> tuple:
    return (
        normalize(crop),
        fuzzy_normalize(name),
        normalize(dtype),
        normalize(stage),
    )


def build_existing_keys(entries: list) -> tuple:
    """Return (exact_set, fuzzy_set) of dedup keys from the NEW_JSON entries."""
    exact = set()
    fuzzy = set()
    for e in entries:
        k  = make_key(e.get("crop",""), e.get("name",""),
                      e.get("diagnosis_type",""), e.get("stage","Field"))
        fk = make_fuzzy_key(e.get("crop",""), e.get("name",""),
                            e.get("diagnosis_type",""), e.get("stage","Field"))
        exact.add(k)
        fuzzy.add(fk)
    return exact, fuzzy


def reorder_entry(entry: dict) -> dict:
    """Return a new dict with keys in KEY_ORDER. Missing keys get empty defaults."""
    defaults = {
        "category"       : "",
        "crop"           : "",
        "diagnosis_type" : "",
        "stage"          : "Field",
        "name"           : "",
        "scientific_name": "",
        "symptoms"       : [],
        "management"     : [],
    }
    merged = {**defaults, **entry}
    return {k: merged[k] for k in KEY_ORDER}


def ensure_list(value) -> list:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def validate(entries: list) -> list:
    """Return list of error strings. Empty list = all good."""
    errors = []
    for i, e in enumerate(entries):
        label = f"[{i}] {e.get('crop','?')}/{e.get('name','?')}"
        if list(e.keys()) != KEY_ORDER:
            errors.append(f"{label} — wrong key order: {list(e.keys())}")
        if e.get("diagnosis_type") not in VALID_DIAGNOSIS_TYPES:
            errors.append(f"{label} — invalid diagnosis_type: '{e.get('diagnosis_type')}'")
        if e.get("stage") not in VALID_STAGES:
            errors.append(f"{label} — invalid stage: '{e.get('stage')}'")
        for arr_field in ("symptoms", "management"):
            val = e.get(arr_field)
            if not isinstance(val, list) or len(val) == 0:
                errors.append(f"{label} — empty/missing '{arr_field}'")
    return errors


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Disease Dataset Merger  (OLD → NEW)")
    print("=" * 65)

    # ── 1. Load files ─────────────────────────────────────────────────────────
    if not os.path.exists(NEW_JSON_PATH):
        print(f"[ERROR] NEW_JSON not found: {NEW_JSON_PATH}")
        sys.exit(1)
    if not os.path.exists(OLD_JSON_PATH):
        print(f"[ERROR] OLD_JSON not found: {OLD_JSON_PATH}")
        sys.exit(1)

    with open(NEW_JSON_PATH, encoding="utf-8") as f:
        new_json = json.load(f)
    with open(OLD_JSON_PATH, encoding="utf-8") as f:
        old_json = json.load(f)

    print(f"\n  NEW_JSON entries loaded : {len(new_json)}")
    print(f"  OLD_JSON entries loaded : {len(old_json)}")

    # ── 2. Ensure all existing NEW_JSON entries are correctly ordered ──────────
    new_json = [reorder_entry(e) for e in new_json]

    # ── 3. Build dedup key sets from NEW_JSON ─────────────────────────────────
    exact_keys, fuzzy_keys = build_existing_keys(new_json)

    # ── 4. Iterate OLD_JSON, collect missing entries ───────────────────────────
    added          = []
    skip_dup       = 0
    skip_non_disease = 0

    for old in old_json:

        # ── Map type ──────────────────────────────────────────────────────────
        old_type_raw = normalize(old.get("type", ""))
        mapped_dtype = TYPE_MAP.get(old_type_raw)
        if mapped_dtype is None:
            skip_non_disease += 1
            continue   # nutrient / abiotic / unknown — skip

        # ── Map category ──────────────────────────────────────────────────────
        old_cat_raw  = normalize(old.get("category", ""))
        mapped_cat   = CATEGORY_MAP.get(old_cat_raw)
        if mapped_cat is None:
            skip_non_disease += 1
            continue   # miscellaneous / general — skip

        # ── Check for duplicates (exact + fuzzy) ──────────────────────────────
        crop     = old.get("crop", "").strip()
        name     = old.get("disease", "").strip()

        ekey = make_key(crop, name, mapped_dtype, "Field")
        fkey = make_fuzzy_key(crop, name, mapped_dtype, "Field")

        if ekey in exact_keys or fkey in fuzzy_keys:
            skip_dup += 1
            continue   # already present in NEW_JSON

        # ── Build new entry ───────────────────────────────────────────────────
        mgmt     = ensure_list(old.get("management", []))
        symptoms = ensure_list(old.get("symptoms",   []))
        sci_name = old.get("causal_organism", "").strip()

        # Fill management if missing
        if not mgmt:
            mgmt = GENERAL_MANAGEMENT.get(mapped_dtype, GENERAL_MANAGEMENT["Fungal"])

        # Title-case crop name for consistency
        crop_title = crop.title()

        new_entry = reorder_entry({
            "category"       : mapped_cat,
            "crop"           : crop_title,
            "diagnosis_type" : mapped_dtype,
            "stage"          : "Field",
            "name"           : name,
            "scientific_name": sci_name,
            "symptoms"       : symptoms,
            "management"     : mgmt,
        })

        # Register to prevent self-duplicates from OLD_JSON
        exact_keys.add(ekey)
        fuzzy_keys.add(fkey)

        added.append(new_entry)

    # ── 5. Merge ──────────────────────────────────────────────────────────────
    merged = new_json + added

    # ── 6. Validate ───────────────────────────────────────────────────────────
    errors = validate(merged)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # ── 8. Report ─────────────────────────────────────────────────────────────
    print(f"\n  Entries added from OLD_JSON       : {len(added)}")
    print(f"  Entries skipped (duplicate)        : {skip_dup}")
    print(f"  Entries skipped (non-disease)      : {skip_non_disease}")
    print(f"  Validation errors                  : {len(errors)}")
    print(f"\n  Final merged total                 : {len(merged)}")
    print(f"  Output written → {OUTPUT_PATH}")

    if errors:
        print("\n  ── VALIDATION ERRORS ───────────────────────────────────")
        for err in errors:
            print(f"    {err}")
    else:
        print("\n  All entries passed validation.")

    # ── 9. Breakdown summaries ────────────────────────────────────────────────
    dtype_counts = Counter(e["diagnosis_type"] for e in merged)
    stage_counts = Counter(e["stage"]          for e in merged)
    cat_counts   = Counter(e["category"]       for e in merged)

    added_cat_counts  = Counter(e["category"] for e in added)
    added_crop_counts = Counter(e["crop"]     for e in added)

    print("\n── By diagnosis_type (merged) ─────────────────────────────")
    for t, c in sorted(dtype_counts.items()):
        print(f"  {t:<12} : {c:>4}")

    print("\n── By stage (merged) ──────────────────────────────────────")
    for s, c in sorted(stage_counts.items()):
        print(f"  {s:<15} : {c:>4}")

    print("\n── By category (merged) ───────────────────────────────────")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:<20} : {cnt:>4}")

    print("\n── Newly added — by category ──────────────────────────────")
    for cat, cnt in sorted(added_cat_counts.items()):
        print(f"  {cat:<20} : {cnt:>4}")

    print("\n── Newly added — top crops ────────────────────────────────")
    for crop, cnt in added_crop_counts.most_common(30):
        print(f"  {crop:<35} : {cnt:>3}")

    print("=" * 65)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
