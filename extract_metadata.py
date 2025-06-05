import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from dateutil.parser import parse as parse_date

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ───────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION & SETUP
# ───────────────────────────────────────────────────────────────────────────

# (a) Paths & folders
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PDF_ROOT_FOLDER = os.path.join(BASE_DIR, "scanned_pdfs")     # will contain subfolders like "SOL Box 11"
CRED_FOLDER     = os.path.join(BASE_DIR, "credentials")
SERVICE_KEY     = os.path.join(CRED_FOLDER, "service_account_key.json")
CHROMA_FOLDER   = os.path.join(BASE_DIR, "chroma_db")

# (b) Google Sheets setup
SPREADSHEET_ID = os.getenv(
    "GOOGLE_SHEET_ID",
    "1ipTfzA5qK8V7BvzuO-hiFCbG50qjhcQP_igndLquEj8"
)

# (c) OpenAI / embeddings / vector store
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
EMBED_MODEL    = "text-embedding-ada-002"

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectordb   = Chroma(
    collection_name="pdf_metadata_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_FOLDER
)

# (d) LLM & RAG prompting setup
llm = ChatOpenAI(model="gpt-4", temperature=0)

TITLE_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You are a metadata extractor. Below are several text passages (from OCR’d scans). 
1) First list any lines that look like a TITLE (e.g., “ALL CAPS HEADINGS” or lines that start with “Title:”). 
   Label them “Candidate 1: …,” “Candidate 2: …,” or write “No obvious candidate.”
2) Then choose the best candidate as the final title. 
   If there is no candidate, answer exactly “UNKNOWN.”

[CONTEXT]
{context}

Respond in two parts:
CANDIDATES:
Candidate 1: …
Candidate 2: …
(…or “No obvious candidate”)
FINAL TITLE: <your chosen title or UNKNOWN>
"""
)

DATE_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
Below is OCR’d text from a scanned document. Identify any date that likely corresponds 
to the document’s creation/publication. Output exactly one date in YYYY/MM/DD (or YYYY/MM or YYYY). 
If you cannot find any date, answer “UNKNOWN.”

[CONTEXT]
{context}

Respond as:
DATE: <YYYY/MM/DD> 
(or “DATE: YYYY/MM” / “DATE: YYYY” / “DATE: UNKNOWN”)
"""
)

DESC_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
Given the OCR’d text passages below, write a brief DESCRIPTION (2–3 sentences max) 
about what this document is about. Focus on factual statements. 
If you cannot find enough information, answer “UNKNOWN.”

[CONTEXT]
{context}

Respond as:
DESCRIPTION: <your 2-3 sentence description> 
(or “DESCRIPTION: UNKNOWN”)
"""
)

title_chain = LLMChain(llm=llm, prompt=TITLE_PROMPT)
date_chain  = LLMChain(llm=llm, prompt=DATE_PROMPT)
desc_chain  = LLMChain(llm=llm, prompt=DESC_PROMPT)

# (e) Text splitter (1,000 tokens per chunk, 200 overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ───────────────────────────────────────────────────────────────────────────
# 2) HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────

def preprocess_page(pil_page):
    """
    1) Convert PIL image → OpenCV BGR
    2) Grayscale → Adaptive threshold → Deskew
    3) Return final binarized & deskewed OpenCV image
    """
    img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=10
    )
    coords = np.column_stack(np.where(binarized > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    (h, w) = binarized.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        binarized, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


def ocr_pdf(pdf_path):
    """
    1) Convert PDF → list of PIL pages at 300 DPI
    2) Preprocess each page, run pytesseract OCR
    3) Return (full_text, avg_confidence%)
    """
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    confidences = []

    for pil_page in pages:
        proc_img = preprocess_page(pil_page)
        data = pytesseract.image_to_data(proc_img, output_type=pytesseract.Output.DICT)
        page_text = pytesseract.image_to_string(proc_img, lang="eng")
        full_text += page_text + "\n\n"

        word_confidences = [
            int(c) for c in data["conf"]
            if isinstance(c, str) and c.isdigit() and int(c) >= 0
        ]
        if word_confidences:
            confidences.append(sum(word_confidences) / len(word_confidences))
        else:
            confidences.append(0)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    return full_text, avg_conf


def split_and_index(pdf_name, full_text):
    """
    1) Split full_text into chunks (~1,000 tokens each)
    2) Embed each chunk and index it into Chroma with metadata
    """
    chunks = splitter.split_text(full_text)
    metadatas = []
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        metadata = {"source_pdf": pdf_name, "chunk_index": i}
        chunk_texts.append(chunk)
        metadatas.append(metadata)

    vectordb.add_texts(chunk_texts, metadatas=metadatas)
    vectordb.persist()


def run_rag_extraction(pdf_name):
    """
    1) For each field (Title, Date, Description), retrieve top-8 similar chunks
    2) Concatenate them, feed to the appropriate LLMChain
    3) Parse the response for the final field value
    4) Return a dict: {"Title": ..., "Date": ..., "Description": ...}
    """
    def _get_context_and_query(query_text):
        docs = vectordb.similarity_search(query_text, k=8)
        passages = [doc.page_content for doc in docs]
        return "\n---\n".join(passages)

    results = {}

    # 1) TITLE
    ctx_title = _get_context_and_query("extract the Title of this document")
    resp_title = title_chain.run(context=ctx_title)
    final_title = "UNKNOWN"
    for line in resp_title.splitlines():
        if line.strip().upper().startswith("FINAL TITLE:"):
            final_title = line.split("FINAL TITLE:", 1)[1].strip()
            break
    results["Title"] = final_title

    # 2) DATE
    ctx_date = _get_context_and_query("find the document’s date (publication or creation)")
    resp_date = date_chain.run(context=ctx_date)
    date_value = "UNKNOWN"
    for part in resp_date.splitlines():
        if part.strip().upper().startswith("DATE:"):
            candidate = part.split("DATE:", 1)[1].strip()
            try:
                _ = parse_date(candidate, fuzzy=False)
                date_value = candidate
            except:
                date_value = "UNKNOWN"
            break
    results["Date"] = date_value

    # 3) DESCRIPTION
    ctx_desc = _get_context_and_query("briefly describe what this document is about")
    resp_desc = desc_chain.run(context=ctx_desc)
    desc_value = "UNKNOWN"
    for line in resp_desc.splitlines():
        if line.strip().upper().startswith("DESCRIPTION:"):
            desc_value = line.split("DESCRIPTION:", 1)[1].strip()
            break
    results["Description"] = desc_value

    return results


# ───────────────────────────────────────────────────────────────────────────
# 3) MAIN PIPELINE
# ───────────────────────────────────────────────────────────────────────────

def fetch_all_tab_names():
    """
    Fetch metadata for the entire spreadsheet, and return a list of tab names.
    """
    creds = Credentials.from_service_account_file(
        SERVICE_KEY,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build("sheets", "v4", credentials=creds)
    spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    all_sheets = spreadsheet.get("sheets", [])
    return [s["properties"]["title"] for s in all_sheets]


def append_rows_to_tab(tab_name, rows):
    """
    Append a list of rows to the given tab at A2.
    Each row is a list of cell values. E.g. ["doc.pdf", "Title", "Desc", "1981/06/15", "87.3%"]
    """
    creds = Credentials.from_service_account_file(
        SERVICE_KEY,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    # Wrap tab name in single quotes (because it contains spaces)
    range_str = f"'{tab_name}'!A2"
    body = {"values": rows}

    sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=range_str,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()


def main():
    # 1) Fetch EVERY tab name from the Google Sheet
    tab_names = fetch_all_tab_names()
    print("Detected tabs in spreadsheet:")
    for t in tab_names:
        print("  •", t)

    # 2) AUTO‐CREATE any missing subfolders under scanned_pdfs/ that match a tab name
    os.makedirs(PDF_ROOT_FOLDER, exist_ok=True)
    for tab in tab_names:
        folder_for_tab = os.path.join(PDF_ROOT_FOLDER, tab)
        if not os.path.isdir(folder_for_tab):
            print(f"🔨 Creating missing folder for tab: {tab!r}")
            os.makedirs(folder_for_tab, exist_ok=True)

    # 3) Now look for subfolders in scanned_pdfs whose names match a tab name
    local_folders = [
        name for name in os.listdir(PDF_ROOT_FOLDER)
        if os.path.isdir(os.path.join(PDF_ROOT_FOLDER, name))
    ]

    # Build a mapping: folder_name (local) → tab_name (in sheets)
    folder_to_tab = {
        folder: folder
        for folder in local_folders
        if folder in tab_names
    }

    if not folder_to_tab:
        print("⚠️  No local folders match any tab names. Exiting.")
        return

    # 4) Process each folder ↔ tab pair
    for folder_name, tab_name in folder_to_tab.items():
        folder_path = os.path.join(PDF_ROOT_FOLDER, folder_name)
        print(f"\n▶ Processing folder “{folder_name}” → tab “{tab_name}”")

        rows_for_tab = []

        # 5) For each PDF in this folder:
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(folder_path, filename)
            print(f"    • OCR & index: {filename}")
            full_text, avg_conf = ocr_pdf(pdf_path)
            split_and_index(filename, full_text)

            # 6) Run RAG extraction
            metadata = run_rag_extraction(filename)

            # 7) Build one row: [PDF Name, Title, Description, Date, OCR Confidence]
            row = [
                filename,
                metadata["Title"],
                metadata["Description"],
                metadata["Date"],
                f"{avg_conf:.1f}%"
            ]
            rows_for_tab.append(row)

        # 8) Append all rows_for_tab to the matching tab
        if rows_for_tab:
            print(f"  ➜ Appending {len(rows_for_tab)} row(s) to tab “{tab_name}” …")
            append_rows_to_tab(tab_name, rows_for_tab)
            print(f"  ✅ Done with “{tab_name}.”")
        else:
            print(f"  ⚠️ No PDFs found in folder “{folder_name}.” Skipping append.")

    print("\n🏁 Pipeline complete.")


if __name__ == "__main__":
    main()
