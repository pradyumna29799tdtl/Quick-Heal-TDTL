import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import re
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#start_time = time.time()

# Constants
IMG_WIDTH = 150
IMG_HEIGHT = 150

# IMG_WIDTH2 = 550
# IMG_HEIGHT2 = 550
max_width = 800
max_height = 600

# Labels for classification
labels = {
    0: "Aadhaar",
    1: "Driving Licence",
    2: "PAN",
    3: "Passport",
    4: "US Passport",
    5: "Voter ID",
    6: "Utility",
}

# Load the model
model_path = 'my_new_model.keras'
cnn_model = load_model(model_path)

# Function to preprocess a single image for classification
def preprocess_image_classification(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    if isinstance(img, str):  # If the input is a file path
        img = image.load_img(img, target_size=(img_width, img_height))
    else:  # If the input is a PIL image (like from a PDF)
        img = img.resize((img_width, img_height))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to perform OCR on the image
def perform_ocr(img):

    oem = 3
    psm = 3
    dpi = 300
    custom_config = f'--psm {psm} --oem {oem}'
    text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
    #print(text)
    return text

# Function to extract information from a document
def extract_info(text, doc_type):
    extracted_info = {}
    patterns = None
    if doc_type == 'PAN':
        patterns = {
            'PAN Number': r'([A-Z]{5}[0-9]{4}[A-Z]{1})',
            'Name': r'Name\s*[:\-]?\s*([A-Za-z\s]+)',
            "Father's Name": r"Father's?\s*Name\s*[:\-]?\s*([A-Za-z\s]+)",
            # 'Date of Birth': r'\d{2}/\d{2}/\d{4}',
        }

    elif doc_type == 'Aadhaar':
        patterns = {
            # 'Aadhaar Number': r'(\d{4} \d{4} \d{4})',
            'Aadhaar Number': r'''(?<!\d)((?:[०-९]{4}[-\s]?[०-९]{4}[-\s]?[०-९]{4})|(?:[௦-௯]{4}[-\s]?[௦-௯]{4}[-\s]?[௦-௯]{4})|(?:[౦-౯]{4}[-\s]?[౦-౯]{4}[-\s]?[౦-౯]{4})|(?:[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}))(?!\d)''',
            
            # 'Name': r'([A-Za-z\s]+)',
            'Date of Birth': r'DOB\s*[:\-]?\s*(\d{2}[-/]\d{2}[-/]\d{4})',
            # 'Gender': r'Male|Female|Other'
        }
        

    elif doc_type == 'Voter ID':
        patterns = {
            'Voter ID': r'\b[A-Za-z]{3}[0-9]{7}\b',
            # 'Name': r"Elector's Name\s*:\s*([A-Za-z\s]+)",
            # "Father's Name": r"Father's Name\s*:\s*([A-Za-z\s]+)",
            # 'Date of Birth': r'Date of Birth\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})',
            # 'Gender': r'Sex\s*:\s*(MALE|FEMALE)',
        }

    elif doc_type == 'Driving Licence':
        patterns = {
            'License Number': r'([A-Z]{2}\d{2} \d{11})',
            # 'Name': r'Name: \s[A-Z]+\s[A-Z]+',
            # 'Date of Birth': r'DOB\s*[:\-]?\s*(\d{2}[-/\.]\d{2}[-/\.]\d{4})',
            # 'Address': r'Address\s*[:\-]?\s*([\w\s,]+)',
        }

    elif doc_type == 'Passport':
        patterns = {
            'Passport Number': r'([A-Z]{1}[0-9]{7})',
            # 'Given Name': r'Given Name\(s\)\s*[:\-]?\s*([A-Z\s]+)',
            # 'Surname': r'Surname\s*[:\-]?\s*([A-Z\s]+)',
            # 'Date of Birth': r'Date of Birth\s*[:\-]?\s*(\d{2}[-/\.]\d{2}[-/\.]\d{4})',
        }

    elif doc_type == 'US Passport':
        patterns = {
            'Passport Number': r'([0-9]{9})',
            # 'Surname': r'Surname\s*[:\-]?\s*([A-Za-z\s]+)',
            # 'Given Name': r'Given Names\s*[:\-]?\s*([A-Za-z\s]+)',
            # 'Date of Birth': r'Fecha de nacimiento\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})',
            # 'Gender': r"Sexo\s*(M|F)"
        }

    if patterns:
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match and match.group(0):
                extracted_info[key] = match.group(0)
        
        for key, value in extracted_info.items():
            if key == 'Name':
                value = value.strip().replace('\n', ' ')
                word = value.split()[1:3] 
                value = ' '.join(word)
            if key == "Father's Name":
                
                value = value.strip().replace('\n', ' ')
                word = value.split()[2:4]
                # print(word)
                value = ' '.join(word)
                # print("word: ",word)
                
            print(f"{key}: {value}")
    
    return extracted_info

# Function to check for keywords and reclassify Utility images
def reclassify_utility(text):
    keywords = {
        'Aadhaar': ['Aadhaar', 'UIDAI'],
        'Driving Licence': ['Driving Licence', 'DL No'],
        'PAN': ['Permanent Account Number', 'Income Tax Department'],
        'Passport': ['Passport', 'Republic of India'],
        'US Passport': ['Passport', 'United States of America'],
        'Voter ID': ['ELECTION','COMMISSION','ELECTION COMMISSION OF INDIA','IDENTITY CARD']
    }

    for doc_type, words in keywords.items():
        for word in words:
            if re.search(word, text, re.IGNORECASE):
                print(f"Keyword '{word}' found for document type '{doc_type}'")
                return doc_type
    
    return 'Utility'

# Function to process PDF and extract images for classification and OCR
def process_pdf_and_extract_images(pdf_path):
    pdf_reader = fitz.open(pdf_path)
    images = []

    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = pdf_reader.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            images.append(image)

    pdf_reader.close()
    return images

# Main function to classify and extract information from the image or PDF
def main(file_path):
    start_time = time.time()
    if file_path.endswith('.pdf'):
        images = process_pdf_and_extract_images(file_path)
        for img in images:
            img_array = preprocess_image_classification(img)
            predictions = cnn_model.predict(img_array)
            predicted_probability = np.max(predictions)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_label = labels[predicted_class[0]] if predicted_probability >= 0.5 else "Utility"
            print(f"Predicted Label: {predicted_label}, Probability: {predicted_probability}")

            text = perform_ocr(img)

            if predicted_label == "Utility":
                new_label = reclassify_utility(text)
                if new_label != "Utility":
                    predicted_label = new_label
                    print(f"Reclassified Label: {predicted_label}")

            extract_info(text, predicted_label)

    else:
        img_array = preprocess_image_classification(image.load_img(file_path))
        predictions = cnn_model.predict(img_array)
        predicted_probability = np.max(predictions)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = labels[predicted_class[0]] if predicted_probability >= 0.5 else "Utility"
        print(f"Predicted Label: {predicted_label}, Probability: {predicted_probability}")

        #img = cv2.imread(file_path)
    with Image.open(file_path) as img: 
         
        width, height = img.size
 
# Check if the image exceeds the threshold
        if width > max_width or height > max_height:
            # Calculate the new size while maintaining the aspect ratio
            scaling_factor = min(max_width / width, max_height / height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
        
            # Rescale the image
            img = img.resize((new_width, new_height),  Image.LANCZOS)
            img.info['dpi'] = (300, 300)  
                #IMG= cv2.resize(img, (IMG_WIDTH2, IMG_HEIGHT2))

        text = perform_ocr(img)

        if predicted_label == "Utility":
            new_label = reclassify_utility(text)
            if new_label != "Utility":
                predicted_label = new_label
                print(f"Reclassified Label: {predicted_label}")

        extract_info(text, predicted_label)

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time Taken: ", time_taken)


if __name__ == "__main__":
    while True:
        file_path = input("Enter the image or PDF file path (or type 'exit' to quit): ")
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.lower() == 'exit':
            break
        main(file_path)