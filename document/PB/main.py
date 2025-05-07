
import fitz
from fitz import open as fitz_open
from docx import Document
from PIL import Image
import pytesseract
import io
import os
import tagging

file_path = input("Please enter file path : ")

if file_path.startswith('"') and file_path.endswith('"'):
    file_path = file_path[1:-1]

file_extension = os.path.splitext(file_path)[1]

if file_extension == ".pdf":
    pdf_reader = fitz.open(file_path)
    extracted_text = ""

    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text = page.get_text()
        if text.strip():
            extracted_text += text
        else:
        # Extract images and apply OCR
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = pdf_reader.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                extracted_text += pytesseract.image_to_string(image, lang='tam+tel+hin+mar+eng+kan')
    pdf_reader.close()


elif file_extension == ".docx":
    # Open the DOCX file in binary mode
    with open(file_path, "rb") as file:
        doc = Document(io.BytesIO(file.read()))
    extracted_text = " ".join(paragraph.text for paragraph in doc.paragraphs)
else:
    print("file is not provided")


# print("tag_pii:", tagging.tag_pii_context("எனது ஆதார் அட்டை எண் ௨௩௭௬ ௫௮௭௦ ௧௭௧௯ ஆகும்."))
print("======================================= Extracted Text =======================================")
print(extracted_text)
print("======================================= Tagged Content =======================================")
print("tag_pii:", tagging.tag_pii_context(extracted_text))
