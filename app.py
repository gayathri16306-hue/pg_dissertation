from flask import Flask, render_template, request, jsonify
import os
import re
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(pdf_path):
    text = ""
    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("pdfplumber error:", e)

    # If no text found, fallback to OCR
    if not text.strip():
        try:
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            print("OCR error:", e)
    return text.strip()

# ---------------- PUBLISHER / FORMAT DETECTION ----------------
def detect_format(text):
    text_lower = text.lower()

    # DOI detection (existing logic)
    doi_match = re.search(r"10\.\d{4,9}/\S+", text.replace("\n", ""))
    if doi_match:
        doi = doi_match.group(0)
        if doi.startswith("10.1109") or "ieee" in text_lower:
            return "IEEE Format", 90
        elif doi.startswith("10.1007") or "springer" in text_lower:
            return "Springer Format", 90
        elif doi.startswith("10.1016") or "elsevier" in text_lower:
            return "Elsevier Format", 90
        elif doi.startswith("10.1145") or "acm" in text_lower:
            return "ACM Format", 90

    # Expanded keyword detection
    if "ieee" in text_lower or "digital object identifier" in text_lower:
        return "IEEE Format", 80
    elif "springer" in text_lower or "springer nature" in text_lower:
        return "Springer Format", 80
    elif "elsevier" in text_lower or "journal homepage" in text_lower:
        return "Elsevier Format", 80
    elif "acm" in text_lower or "association for computing machinery" in text_lower:
        return "ACM Format", 80
    elif "engineering and science" in text_lower or "e&s" in text_lower:
        return "E&S Format", 70
    elif "physicae organum" in text_lower or "universidade de brasília" in text_lower:
        # Detect Physicae Organum
        return "Physicae Organum Format", 85
    else:
        return "Unknown / Custom Format", 0


# ---------------- SIMILARITY CALCULATION ----------------
def compute_similarity(texts):
    if len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded_files = request.files.getlist('pdfs')
    results, texts, filenames = [], [], []

    for file in uploaded_files:
        if file and file.filename.lower().endswith('.pdf'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            text = extract_text_from_pdf(path)
            paper_format, confidence = detect_format(text)
            results.append({
                'filename': file.filename,
                'format': paper_format,
                'confidence': f"{confidence}%"
            })
            texts.append(text)
            filenames.append(file.filename)

    # Compute similarity
    sim_data = []
    if len(texts) > 1:
        sim_matrix = compute_similarity(texts)
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                sim_data.append({
                    'file1': filenames[i],
                    'file2': filenames[j],
                    'similarity': f"{sim_matrix[i][j]*100:.2f}%"
                })

    return jsonify({'results': results, 'similarity': sim_data})

# ---------------- APP ENTRY POINT ----------------
if __name__ == '__main__':
    # If Tesseract is not in PATH, set it here
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    app.run(debug=True)
