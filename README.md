# Invoice OCR & AI Extraction

A powerful Streamlit application that extracts structured data from invoice PDFs and images using OCR (Tesseract) and AI (OpenAI or Cohere) for intelligent field extraction.

## Features

- 📄 **Multi-format Support**: Process PDF, PNG, JPG, JPEG, and TIFF files
- 🔍 **High-quality OCR**: Tesseract-powered text extraction with configurable DPI (300 default)
- 🤖 **AI Extraction**: Support for both OpenAI and Cohere models for structured data extraction
- 📊 **Multiple Export Formats**: Download results as JSON, CSV, or Excel (with multiple sheets)
- 📈 **Progress Tracking**: Real-time progress bars for OCR processing
- 🎨 **Clean UI**: Modern, intuitive interface with metrics and tabbed views
- 💾 **Comprehensive Output**: Header fields, line items, and flattened data views

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR installed on your system
  - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:

   **For OpenAI version** (`main_openai.py`):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   TESSERACT_LANG=eng
   TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe  # Optional, if not in PATH
   ```

   **For Cohere version** (`main.py`):
   ```env
   COHERE_API_KEY=your_cohere_api_key_here
   COHERE_MODEL=command-r-plus
   TESSERACT_LANG=eng
   TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe  # Optional
   ```

## Usage

### Run with OpenAI:
```powershell
streamlit run main_openai.py
```

### Run with Cohere:
```powershell
streamlit run main.py
```

### Using the Application

1. **Upload Invoice**: Click the file uploader and select your invoice (PDF or image)
2. **Optional**: Enter a currency hint (e.g., "INR", "USD")
3. **View Results**:
   - **OCR Preview**: See the extracted text
   - **Structured JSON**: Complete invoice data in JSON format
   - **Header Fields**: Invoice metadata (vendor, dates, amounts, etc.)
   - **Flat Rows**: Tabular view combining header + line items
   - **Downloads**: Export as CSV, Excel, or JSON

## Extracted Fields

The application extracts the following invoice fields:

### Header Information
- `vendor_name` - Vendor/supplier name
- `vendor_address` - Vendor address
- `vendor_gstin` - Vendor GST/Tax ID
- `invoice_number` - Invoice number
- `invoice_date` - Invoice date (ISO-8601 format)
- `due_date` - Payment due date
- `buyer_name` - Buyer/customer name
- `buyer_address` - Buyer address
- `buyer_gstin` - Buyer GST/Tax ID
- `currency` - Currency code (INR, USD, etc.)
- `subtotal` - Subtotal amount
- `tax_percent` - Tax percentage
- `tax_amount` - Tax amount
- `total` - Total amount
- `notes` - Additional notes

### Line Items
Each line item includes:
- `description` - Item description
- `quantity` - Quantity
- `unit_price` - Price per unit
- `amount` - Total amount for line item

## Project Structure

```
invoice/
├── main.py                 # Cohere version
├── main_openai.py          # OpenAI version
├── main1.py               # Alternative implementations
├── main2.py
├── mainn.py
├── test.py
├── app.py
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
└── README.md              # This file
```

## Dependencies

- `streamlit` - Web application framework
- `pandas` - Data manipulation and CSV export
- `pydantic` - Data validation
- `Pillow` - Image processing
- `pytesseract` - Python wrapper for Tesseract OCR
- `openai` - OpenAI API client
- `cohere` - Cohere API client
- `PyMuPDF` - PDF processing (fitz)
- `python-dotenv` - Environment variable management
- `openpyxl` - Excel file generation

## API Keys

### OpenAI
Get your API key from: https://platform.openai.com/api-keys

Recommended models:
- `gpt-4o-mini` (cost-effective, fast)
- `gpt-4o` (more accurate, higher cost)
- `gpt-4-turbo`

### Cohere
Get your API key from: https://dashboard.cohere.com/api-keys

Recommended models:
- `command-r-plus` (best for complex extraction)
- `command-r`

## Troubleshooting

### Tesseract Not Found
If you get a tesseract error:
1. Install Tesseract OCR
2. Set `TESSERACT_PATH` in `.env` to the full path of `tesseract.exe`

### Network/Connection Errors
If pip installation fails:
1. Check your internet connection
2. Try upgrading pip: `python -m pip install --upgrade pip`
3. Use a different network or VPN if behind a firewall

### Low OCR Accuracy
- Ensure invoice images are clear and high-resolution
- Try different `TESSERACT_LANG` settings (e.g., `eng+fra` for multiple languages)
- Increase DPI in the code if needed

### AI Extraction Issues
- Verify API keys are correct in `.env`
- Check API quota/credits
- Try different models
- Ensure OCR text quality is good

## Performance Tips

- **High-quality scans**: Use 300 DPI or higher for best OCR results
- **PDF optimization**: Multi-page PDFs are processed page-by-page with caching
- **Model selection**: 
  - Use `gpt-4o-mini` for cost-effectiveness
  - Use `gpt-4o` or `command-r-plus` for complex invoices

## License

This project is provided as-is for educational and commercial use.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the `.env` configuration
3. Ensure all dependencies are installed correctly
4. Verify Tesseract is properly installed

---

**Note**: This application requires active API keys for OpenAI or Cohere. API usage will incur costs based on your provider's pricing.
