import json
import re

INPUT_FILE = "cereals_diseases.txt"
OUTPUT_FILE = "cereals_diseases.json"

CATEGORY = "Cereal"
CROP = "Rice"
DIAGNOSIS_TYPE = "Disease"


def normalize_text(text):
    text = text.replace("\uf0b7", "•")
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_pages(text):
    # Your pasted text seems to repeat this header
    parts = re.split(r"\bCereals\s*::\s*Rice\b", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def remove_noise(page):
    # remove junk sections at end
    stop_headers = [
        "Source of information",
        "Acknowledgements",
        "Acknowledgment",
        "References",
        "Reference"
    ]
    for header in stop_headers:
        page = re.split(rf"\n\s*{re.escape(header)}\s*:?", page, flags=re.IGNORECASE)[0]

    # remove repeated crop headers if still present
    page = re.sub(r"Agricultural crops\s*::\s*Cereals\s*::\s*Rice", "", page, flags=re.IGNORECASE)
    page = re.sub(r"\bCereals\s*::\s*Rice\b", "", page, flags=re.IGNORECASE)

    return page.strip()


def clean_bullet_line(line):
    line = re.sub(r"^[•\-\*\u2022]+\s*", "", line).strip()
    return line


def is_heading(line):
    t = line.strip().rstrip(":").strip().lower()
    headings = {
        "symptoms",
        "management",
        "favourable conditions",
        "favorable conditions",
        "favourable conditions / epidemiology",
        "favorable conditions / epidemiology",
        "pathogen",
        "pathogen character",
        "bacterium",
        "identification of pathogen",
        "causal organism",
        "vector",
        "other management",
        "survival and mode of spread",
        "mode of spread and survival",
    }
    return t in headings


def extract_title_block(page):
    lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
    title_lines = []

    for line in lines:
        low = line.lower()

        if "other management" in low:
            continue

        if is_heading(line):
            break

        title_lines.append(line)

    return title_lines


def extract_name_and_scientific(page):
    title_lines = extract_title_block(page)

    if not title_lines:
        return "Unknown", ""

    # remove crop headers if they survived
    filtered = []
    for line in title_lines:
        low = line.lower().strip()
        if low in {"cereals :: rice", "agricultural crops :: cereals :: rice"}:
            continue
        filtered.append(line)

    title_lines = filtered
    if not title_lines:
        return "Unknown", ""

    # Pattern 1:
    # Bacterial leaf blight: Causal organism: Xanthomonas ...
    for line in title_lines:
        m = re.match(r"^(.*?)\s*:\s*Causal organism\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()

    # Pattern 2:
    # False smut / Lakshmi disease
    # Causal organism: Ustilaginoidea virens ...
    for i, line in enumerate(title_lines):
        if re.match(r"^Causal organism\s*:", line, flags=re.IGNORECASE):
            sci = re.sub(r"^Causal organism\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            prev = title_lines[i - 1].strip() if i > 0 else "Unknown"
            return prev, sci

    # Pattern 3:
    # Rice grassy stunt disease : Rice grassy stunt virus
    # Vector: Brown plant hopper ...
    for i, line in enumerate(title_lines):
        if re.match(r"^Vector\s*:", line, flags=re.IGNORECASE):
            prev = title_lines[i - 1].strip() if i > 0 else "Unknown"
            if ":" in prev:
                left, right = prev.split(":", 1)
                return left.strip(), right.strip()
            return prev, ""

    # Pattern 4:
    # Blast : Magnaporthe oryzae ...
    for line in title_lines:
        if ":" in line and "vector" not in line.lower():
            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip()

            if left and len(left.split()) <= 12 and left.lower() not in {
                "survival and mode of spread",
                "mode of spread and survival",
            }:
                return left, right

    # fallback: first short title-like line
    for line in title_lines:
        if len(line.split()) <= 12 and not is_heading(line):
            return line.strip(), ""

    return title_lines[0].strip(), ""


def find_sections(page):
    lines = [ln.rstrip() for ln in page.split("\n")]
    sections = []

    valid_sections = {
        "symptoms",
        "management",
        "favourable conditions",
        "favorable conditions",
        "favourable conditions / epidemiology",
        "favorable conditions / epidemiology",
        "pathogen",
        "pathogen character",
        "bacterium",
        "identification of pathogen",
        "survival and mode of spread",
        "mode of spread and survival",
    }

    for i, line in enumerate(lines):
        t = line.strip().rstrip(":").strip().lower()
        if t in valid_sections:
            sections.append((t, i))

    return lines, sections


def get_section_content(lines, sections, target_names):
    target_names = {x.lower() for x in target_names}

    for idx, (name, start_line) in enumerate(sections):
        if name in target_names:
            end_line = len(lines)
            if idx + 1 < len(sections):
                end_line = sections[idx + 1][1]

            block_lines = lines[start_line + 1:end_line]
            cleaned = []

            for ln in block_lines:
                x = clean_bullet_line(ln)
                if not x:
                    continue
                cleaned.append(x)

            return cleaned

    return []


def generalize_management(lines):
    out = []
    for line in lines:
        low = line.lower()

        if "resistant variet" in low or "moderately resistant variet" in low:
            out.append("Use region-specific resistant varieties.")
            continue

        line = re.sub(r"\bCO\s*\d+\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\bADT\s*\d+\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\bASD\s*\d+\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\bIR\s*\d+\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s{2,}", " ", line).strip(" ,;.")
        if line:
            if not line.endswith("."):
                line += "."
            out.append(line)

    return out


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw = f.read()

raw = normalize_text(raw)
pages = split_pages(raw)

results = []

for page in pages:
    page = remove_noise(page)
    if not page:
        continue

    name, scientific_name = extract_name_and_scientific(page)
    lines, sections = find_sections(page)

    symptoms = get_section_content(lines, sections, ["symptoms"])

    risk_factors = get_section_content(
        lines,
        sections,
        [
            "favourable conditions",
            "favorable conditions",
            "favourable conditions / epidemiology",
            "favorable conditions / epidemiology",
        ],
    )

    management = get_section_content(lines, sections, ["management"])

    entry = {
        "category": CATEGORY,
        "crop": CROP,
        "diagnosis_type": DIAGNOSIS_TYPE,
        "name": name,
        "scientific_name": scientific_name,
        "symptoms": symptoms,
        "risk_factors": risk_factors,
        "management": generalize_management(management),
    }

    if entry["name"] != "Unknown":
        results.append(entry)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done! Created {OUTPUT_FILE} with {len(results)} entries.")