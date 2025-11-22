import os
import io
import json
from pathlib import Path
from typing import List, Optional, Callable

import streamlit as st
import pandas as pd
from pydantic import BaseModel, ValidationError, Field
from PIL import Image
import pytesseract
import cohere
import fitz
from dotenv import load_dotenv

load_dotenv()

def _maybe_set_tesseract_path():
    tpath = os.getenv("TESSERACT_PATH")
    if tpath and Path(tpath).exists():
        pytesseract.pytesseract.tesseract_cmd = tpath

_maybe_set_tesseract_path()

class LineItem(BaseModel):
    description: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None

class InvoiceSchema(BaseModel):
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_gstin: Optional[str] = Field(None, alias="vendor_gst")
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    buyer_gstin: Optional[str] = Field(None, alias="buyer_gst")
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    tax_percent: Optional[float] = None
    tax_amount: Optional[float] = None
    total: Optional[float] = None
    line_items: List[LineItem] = []
    notes: Optional[str] = None

@st.cache_data(show_spinner=False)
def pdf_to_images(file_bytes: bytes, dpi: int = 300):
    doc = fitz.open("pdf", file_bytes)
    images = []
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        if pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

@st.cache_data(show_spinner=False)
def ocr_image(img: Image.Image, lang: str):
    return pytesseract.image_to_string(img, lang=lang)

def ocr_any(file_bytes: bytes, file_type: str, lang: str, on_progress: Optional[Callable[[int, int], None]] = None):
    texts = []
    if file_type == "pdf":
        pages = pdf_to_images(file_bytes)
        total = len(pages)
        for i, p in enumerate(pages, start=1):
            texts.append(ocr_image(p.convert("RGB"), lang))
            if on_progress:
                on_progress(i, total)
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        texts.append(ocr_image(img, lang))
        if on_progress:
            on_progress(1, 1)
    return "\n".join(texts)

def build_prompt(text: str, currency_hint: Optional[str] = None):
    schema = {
        "vendor_name": "",
        "vendor_address": "",
        "vendor_gstin": "",
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "buyer_name": "",
        "buyer_address": "",
        "buyer_gstin": "",
        "currency": currency_hint or "",
        "subtotal": None,
        "tax_percent": None,
        "tax_amount": None,
        "total": None,
        "line_items": [
            {"description": "", "quantity": None, "unit_price": None, "amount": None}
        ],
        "notes": ""
    }
    examples = [
        {
            "input": "INVOICE\nNo: INV-104\nDate: 2024-04-02\nVendor: Acme Tools Pvt Ltd GSTIN 27ABCDE1234F1Z5\nBill To: XYZ Industries GSTIN 07PQRSX5678L1Z2\nItem Hammer 2 x 150.00 300.00\nSubtotal 300.00\nGST 18% 54.00\nTotal INR 354.00",
            "output": {
                "vendor_name": "Acme Tools Pvt Ltd",
                "vendor_gstin": "27ABCDE1234F1Z5",
                "invoice_number": "INV-104",
                "invoice_date": "2024-04-02",
                "buyer_name": "XYZ Industries",
                "buyer_gstin": "07PQRSX5678L1Z2",
                "currency": "INR",
                "subtotal": 300.0,
                "tax_percent": 18.0,
                "tax_amount": 54.0,
                "total": 354.0,
                "line_items": [
                    {"description": "Hammer", "quantity": 2, "unit_price": 150.0, "amount": 300.0}
                ]
            }
        }
    ]
    instruction = (
        "Extract an invoice as strict JSON in the target schema. "
        "Infer numbers as floats, dates in ISO-8601 (YYYY-MM-DD), currency codes like INR/USD if present. "
        "Return only JSON. If a field is unknown, use null or empty string. "
        "Combine multi-line addresses into a single line separated by commas. "
        "Do not hallucinate line items."
    )
    payload = {"instruction": instruction, "schema": schema, "examples": examples, "text": text}
    return json.dumps(payload)

def cohere_extract(text: str, model: str, api_key: str, currency_hint: Optional[str] = None):
    client = cohere.Client(api_key)
    prompt = build_prompt(text, currency_hint)
    resp = client.generate(
        model=model,
        prompt=("You are a JSON information extractor.\n"
                "Given JSON with instruction, schema, examples, and text, extract fields.\n"
                + prompt + "\nReturn only JSON."),
        max_tokens=1200,
        temperature=0,
    )
    out = resp.generations[0].text.strip()
    if out.startswith("```"):
        try:
            out = out.split("```", 2)[1].strip()
        except Exception:
            out = out.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(out)
    except Exception:
        try:
            start = out.find("{")
            end = out.rfind("}")
            data = json.loads(out[start:end+1])
        except Exception:
            data = {}
    try:
        parsed = InvoiceSchema.model_validate(data)
        return json.loads(parsed.model_dump_json())
    except ValidationError:
        return data

def to_flat_table(data: dict) -> pd.DataFrame:
    header = {k: v for k, v in data.items() if k != "line_items"}
    items = data.get("line_items") or []
    if not items:
        return pd.DataFrame([header])
    rows = [{**header, **(it if isinstance(it, dict) else {})} for it in items]
    df = pd.DataFrame(rows)
    header_order = [
        "vendor_name","vendor_address","vendor_gstin",
        "buyer_name","buyer_address","buyer_gstin",
        "invoice_number","invoice_date","due_date",
        "currency","subtotal","tax_percent","tax_amount","total","notes"
    ]
    item_order = ["description","quantity","unit_price","amount"]
    present_header = [c for c in header_order if c in df.columns]
    present_items = [c for c in item_order if c in df.columns]
    middle = [c for c in df.columns if c not in present_header + present_items]
    return df[present_header + middle + present_items]

def export_invoice_json_to_excel(data: dict) -> bytes:
    header = {k: v for k, v in data.items() if k != "line_items"}
    header_df = pd.DataFrame(list(header.items()), columns=["Field", "Value"])
    items_df = pd.DataFrame(data.get("line_items") or [])
    flat_df = to_flat_table(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        header_df.to_excel(writer, sheet_name="Invoice_Header", index=False)
        if not items_df.empty:
            keep = [c for c in ["description", "quantity", "unit_price", "amount"] if c in items_df.columns]
            (items_df[keep] if keep else items_df).to_excel(writer, sheet_name="Line_Items", index=False)
        flat_df.to_excel(writer, sheet_name="Invoice_Flat", index=False)
    buf.seek(0)
    return buf.getvalue()

st.set_page_config(page_title="Invoice OCR + Cohere", layout="wide")
st.markdown(
    """
    <style>
      .app-header {display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:0.25rem 0;}
      .pill {font-size:12px; padding:4px 10px; border-radius:999px; background:#f1f5f9; border:1px solid #e2e8f0; display:inline-block;}
      .card {border:1px solid #e5e7eb; border-radius:12px; padding:1rem; background:#fff;}
      .muted {color:#64748b;}
      .downloads {display:flex; gap:.5rem; flex-wrap:wrap;}
    </style>
    """,
    unsafe_allow_html=True
)

api_key = os.getenv("COHERE_API_KEY", "")
model = os.getenv("COHERE_MODEL", "")
ocr_lang = os.getenv("TESSERACT_LANG", "")

st.markdown('<div class="app-header"><h1 style="margin:0;">Invoice Reader</h1><span class="pill">OCR + Cohere JSON</span></div>', unsafe_allow_html=True)
with st.container():
    cols = st.columns([3, 2])
    with cols[0]:
        file = st.file_uploader("Upload invoice (PDF/Image)", type=["pdf", "png", "jpg", "jpeg", "tiff"])
    with cols[1]:
        st.selectbox("Mode", ["Extract JSON"], index=0, key="mode_select")
        st.text_input("Currency hint (optional)", value="", key="currency_hint")

status_cols = st.columns(3)
with status_cols[0]:
    st.metric("Cohere Model", model if model else "Not set")
with status_cols[1]:
    st.metric("OCR Lang", ocr_lang if ocr_lang else "Not set")
with status_cols[2]:
    st.metric("Tesseract", "Configured" if pytesseract.pytesseract.tesseract_cmd else "Default")

if not api_key:
    st.info("Set COHERE_API_KEY in .env")
if not model:
    st.info("Set COHERE_MODEL in .env (e.g., command-r-plus)")
if not ocr_lang:
    st.info("Set TESSERACT_LANG in .env (e.g., eng)")

raw_text = ""
metadata = {}
if file is not None:
    kind = "pdf" if file.type == "application/pdf" else "image"
    try:
        if kind == "pdf":
            doc = fitz.open(stream=file.getvalue(), filetype="pdf")
            metadata = {"name": file.name, "type": "PDF", "size_kb": round(len(file.getvalue())/1024, 1), "pages": doc.page_count}
        else:
            metadata = {"name": file.name, "type": file.type.split("/")[-1].upper(), "size_kb": round(len(file.getvalue())/1024, 1), "pages": 1}
    except Exception:
        metadata = {"name": file.name, "type": "Unknown", "size_kb": round(len(file.getvalue())/1024, 1), "pages": "-"}

    with st.expander("File details", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**Name**\n\n{metadata.get('name','-')}")
        c2.write(f"**Type**\n\n{metadata.get('type','-')}")
        c3.write(f"**Size (KB)**\n\n{metadata.get('size_kb','-')}")
        c4.write(f"**Pages**\n\n{metadata.get('pages','-')}")

    if ocr_lang and api_key and model:
        progress = st.progress(0.0, text="Starting OCR…")
        def on_progress(done, total):
            pct = done / max(total, 1)
            progress.progress(pct, text=f"OCR {done}/{total} pages")

        with st.status("Running OCR", expanded=False) as s:
            raw_text = ocr_any(file.read(), kind, ocr_lang, on_progress=on_progress)
            progress.progress(1.0, text="OCR complete")
            s.update(label="OCR complete", state="complete")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("OCR Preview")
            st.text_area("Text", raw_text[:20000], height=420)

        with st.status("Extracting structured JSON with Cohere", expanded=False) as s2:
            data = cohere_extract(raw_text, model, api_key, st.session_state.get("currency_hint") or None)
            s2.update(label="Extraction complete", state="complete")

        tabs = st.tabs(["Structured JSON", "Header Fields", "Flat Rows", "Downloads"])
        with tabs[0]:
            st.code(json.dumps(data, ensure_ascii=False, indent=2))
        with tabs[1]:
            meta = {k: v for k, v in data.items() if k not in ("line_items",)}
            if meta:
                st.json(meta)
            else:
                st.write("No header fields parsed.")
        with tabs[2]:
            flat = to_flat_table(data)
            if not flat.empty:
                st.dataframe(flat, use_container_width=True)
            else:
                st.write("No rows to display.")
        with tabs[3]:
            area = st.container()
            with area:
                if 'flat' not in locals():
                    flat = to_flat_table(data)
                csv_bytes = flat.to_csv(index=False).encode("utf-8") if not flat.empty else "".encode("utf-8")
                xlsx_bytes = export_invoice_json_to_excel(data)
                st.markdown('<div class="downloads">', unsafe_allow_html=True)
                st.download_button("CSV (Flat)", data=csv_bytes, file_name="invoice_flat.csv", disabled=flat.empty)
                st.download_button("Excel (All Sheets)", data=xlsx_bytes, file_name="invoice_output.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.download_button("JSON", data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
                                   file_name="invoice.json")
                st.markdown('</div>', unsafe_allow_html=True)

        kpi_cols = st.columns(4)
        try:
            li = data.get("line_items") or []
            kpi_cols[0].metric("Line items", len(li))
        except Exception:
            kpi_cols[0].metric("Line items", 0)
        kpi_cols[1].metric("Subtotal", str(data.get("subtotal") if isinstance(data, dict) else "-"))
        kpi_cols[2].metric("Tax %", str(data.get("tax_percent") if isinstance(data, dict) else "-"))
        kpi_cols[3].metric("Total", str(data.get("total") if isinstance(data, dict) else "-"))

        st.divider()
        util_cols = st.columns([1,1,1])
        with util_cols[0]:
            if st.button("Clear"):
                st.cache_data.clear()
                st.experimental_rerun()
        with util_cols[1]:
            st.write("")
        with util_cols[2]:
            st.caption("High-DPI PDF rasterization enabled")

    else:
        st.warning("Configure .env first: COHERE_API_KEY, COHERE_MODEL, TESSERACT_LANG")

st.markdown("---")
st.caption("Clean UI • High-DPI PDF rasterization • Progress-aware OCR • Cohere JSON extraction • CSV/Excel exports")
