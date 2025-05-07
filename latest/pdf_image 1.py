# import fitz  # PyMuPDF
# from PIL import Image
# import pytesseract
# import io
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import re
# import time
# import os
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import tensorflow as tf
# import sys
# from pymongo import MongoClient
# tf.config.optimizer.set_jit(True)  # Enable XLA


# # Environment settings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # Constants
# IMG_WIDTH = 150
# IMG_HEIGHT = 150
# MAX_WIDTH = 800
# MAX_HEIGHT = 600
# DPI = 300


# client = MongoClient('localhost', 27017)
# db = client['image_recog']  

# labels = {
#     0 : "Aadhaar",
#     1 : "Canada ID",
#     2 : "Czeh ID",
#     3 : "Denmark PID",
#     4 : "Driving Licence",
#     5 : "Finland ID",
#     6 : "Israel NID",
#     7 : "PAN",
#     8 : "Passport",
#     9 : "Polan ID",
#     10 : "Singapore ID",
#     11 : "South Africa ID",
#     12 : "Spain SSN",
#     13 : "UK DL",
#     14 : "US Passport",
#     15 : "Utility",
#     16 : "Voter ID"
# }

# def resource_path(relative_path):
#     """
#     Get the absolute path to the resource.

#     This function handles both development and PyInstaller bundle scenarios.

#     Parameters:
#         relative_path (str): The relative path to the resource.

#     Returns:
#         str: The absolute path to the resource.
#     """
#     if hasattr(sys, '_MEIPASS'):
#         return os.path.join(sys._MEIPASS, relative_path)
#     return os.path.join(os.path.abspath("."), relative_path)


# model_path = resource_path('foreign_indian_classificiation_model.keras')

# # Load the model
# cnn_model = load_model(model_path)

# def preprocess_image_classification(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
#     """
#     Preprocess an image for classification.

#     This function converts the image to RGB if needed, resizes it, and normalizes it.

#     Parameters:
#         img (PIL.Image): The image to preprocess.
#         img_width (int): The target width of the image.
#         img_height (int): The target height of the image.

#     Returns:
#         np.array: The preprocessed image array.
#     """
#     if img.mode == 'RGBA':
#         img = img.convert('RGB')
#     elif img.mode != 'RGB':
#         img = img.convert('RGB')
        
#     img = img.resize((img_width, img_height))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]
#     return img_array


# # Function to save extracted information to MongoDB
# def save_to_db(file_path, extract_info, db):
#     data = {
#         "image_path": file_path,
#         #"label": predicted_label,
#         "extracted_info": extract_info
#     }
#     db.abcd.insert_one(data)
#     print(f"Information from {file_path} has been saved to the database")



# def perform_ocr(img):
#     """
#     Perform Optical Character Recognition (OCR) on an image.

#     This function uses Tesseract to extract text from the image.

#     Parameters:
#         img (PIL.Image): The image to perform OCR on.

#     Returns:
#         str: The extracted text from the image.
#     """
#     custom_config = '--psm 11 --oem 3'
#     return pytesseract.image_to_string(img, lang='eng', config=custom_config)

# def extract_info(text, doc_type):
#     """
#     Extract specific information from the text based on the document type.

#     This function uses regular expressions to find and extract patterns such as PAN numbers or Aadhaar numbers.

#     Parameters:
#         text (str): The text to extract information from.
#         doc_type (str): The type of document (e.g., 'PAN', 'Aadhaar').

#     Returns:
#         dict: A dictionary of extracted information.
#     """
#     patterns = {
#         'PAN': {'PAN Number': r'([A-Z]{5}[0-9]{4}[A-Z]{1})'},
#         'Aadhaar': {'Aadhaar Number': r'''(?<!\d)((?:[०-९]{4}[-\s]?[०-९]{4}[-\s]?[०-९]{4})|(?:[௦-௯]{4}[-\s]?[௦-௯]{4}[-\s]?[௦-௯]{4})|(?:[౦-౯]{4}[-\s]?[౦-౯]{4}[-\s]?[౦-౯]{4})|(?:[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}))(?!\d)'''},
#         'Voter ID': {'Voter ID': r'\b[A-Za-z]{3}[0-9]{7}\b'},
#         'Driving Licence': {'License Number': r'([A-Z]{2}\d{2} \d{11})'},
#         'Passport': {'Passport Number': r'([A-Z]{1}[0-9]{7})'},
#         'US Passport': {'Passport Number': r'([0-9]{9})'},
#         'Canada ID':{'ID Number': r'(\d{4}-\d{4})'},
#         'Czeh ID':{},
#         'Denmark PID':{'ID Number': r'\d{6}-\d{4}'},
#         'Finland ID':{'ID Number': r'([0-9]{10})'},
#         'Israel NID':{ 'ID Number': r'([0-9]{9})'},
#         'Polan ID':{'ID Number': r'\d{2}-\d{4}-\d{4}-\d{2}'},
#         'Singapore ID':{'ID Number': r'[S][0-9]{7}[A-Z]'},
#         'South Africa ID':{'ID Number': r'([0-9]{13})'},
#         'Spain SSN':{'ID Number':r"\b\d{11}\b"},
#         'UK DL':{'ID Number': r'[A-Z]{2}\d{6}[A-Z]{2}'},
#     }.get(doc_type, {})

#     extracted_info = {key: match.group(0) for key, pattern in patterns.items() if (match := re.search(pattern, text))}
#     for key, value in extracted_info.items():
#         print(f"{key}: {value}")
    
#     return extracted_info


# def reclassify_utility(text):
#     """
#     Reclassify images labeled as 'Utility' based on specific keywords found in the text.

#     This function searches for keywords associated with different document types to reclassify Utility images.

#     Parameters:
#         text (str): The text to search for keywords.

#     Returns:
#         str: The reclassified document type, or 'Utility' if no keywords are found.
#     """
#     keywords = {
#         'Aadhaar': ['Aadhaar', 'UIDAI'],
#         'Driving Licence': ['Driving Licence', 'DL No'],
#         'PAN': ['Permanent Account Number', 'Income Tax Department'],
#         'Passport': ['Passport', 'Republic of India'],
#         'US Passport': ['Passport', 'United States of America'],
#         'Voter ID': ['ELECTION','COMMISSION','ELECTION COMMISSION OF INDIA','IDENTITY CARD']
#     }

#     for doc_type, words in keywords.items():
#         if any(re.search(word, text, re.IGNORECASE) for word in words):
#             print(f"Keyword found for document type '{doc_type}'")
#             return doc_type
#     return 'Utility'


# def process_pdf_and_extract_images(pdf_path):
#     """
#     Process a PDF file and extract images from it.

#     This function uses PyMuPDF to extract images from each page of the PDF.

#     Parameters:
#         pdf_path (str): The path to the PDF file.

#     Returns:
#         list: A list of extracted images as PIL.Image objects.
#     """
#     pdf_reader = fitz.open(pdf_path)
#     images = []

#     for page_num in range(len(pdf_reader)):
#         page = pdf_reader.load_page(page_num)
#         for img_info in page.get_images(full=True):
#             xref = img_info[0]
#             base_image = pdf_reader.extract_image(xref)
#             img = Image.open(io.BytesIO(base_image["image"])).convert("RGBA")
#             images.append(img)

#     pdf_reader.close()
#     return images


# def resize_image(img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
#     """
#     Resize an image while maintaining its aspect ratio.

#     This function resizes the image so that it fits within the specified maximum width and height.

#     Parameters:
#         img (PIL.Image): The image to resize.
#         max_width (int): The maximum width of the resized image.
#         max_height (int): The maximum height of the resized image.

#     Returns:
#         PIL.Image: The resized image.
#     """
#     width, height = img.size
#     if width > max_width or height > max_height:
#         scaling_factor = min(max_width / width, max_height / height)
#         new_size = (int(width * scaling_factor), int(height * scaling_factor))
#         img = img.resize(new_size, Image.LANCZOS)
#         img.info['dpi'] = (300, 300)  
#     return img


# def process_single_image(img):
#     """
#     Process a single image for classification and information extraction.

#     This function preprocesses the image, predicts its class using the CNN model, performs OCR to extract text, 
#     and reclassifies Utility images based on the text content.

#     Parameters:
#         img (PIL.Image): The image to process.
#     """
#     img_array = preprocess_image_classification(img)
#     predictions = cnn_model.predict(img_array)
#     predicted_probability = np.max(predictions)
#     predicted_class = np.argmax(predictions, axis=1)
#     predicted_label = labels[predicted_class[0]] if predicted_probability >= 0.5 else "Utility"
#     print(f"Predicted Label: {predicted_label}, Probability: {predicted_probability}")

#     img = resize_image(img)
#     text = perform_ocr(img)

#     if predicted_label == "Utility":
#         new_label = reclassify_utility(text)
#         if new_label != "Utility":
#             predicted_label = new_label
#             print(f"Reclassified Label: {predicted_label}")

#     extract_info(text, predicted_label)


# def main(file_path):
#     """
#     Main function to classify and extract information from an image or PDF.

#     This function handles both image and PDF files, processes each image found, and uses threading for parallel processing.

#     Parameters:
#         file_path (str): The path to the image or PDF file.
#     """
#     start_time = time.time()

#     if file_path.endswith('.pdf'):
#         images = process_pdf_and_extract_images(file_path)
#     else:
#         images = [Image.open(file_path)]

#     # Using ThreadPoolExecutor to process images in parallel
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_single_image, img) for img in images]
#         for future in as_completed(futures):
#             future.result()  # This will raise any exceptions encountered during processing
    
#     save_to_db(file_path,extract_info, db)

#     print(f"Time Taken: {time.time() - start_time:.2f} seconds")


# if __name__ == "__main__":
#     while True:
#         file_path = input("Enter the image or PDF file path (or type 'exit' to quit): ")
#         if file_path.startswith('"') and file_path.endswith('"'):
#             file_path = file_path[1:-1]
#         if file_path.lower() == 'exit':
#             break
#         main(file_path)





import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
import sys
# from pymongo import MongoClient

tf.config.optimizer.set_jit(True)  # Enable XLA

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Constants
IMG_WIDTH = 150
IMG_HEIGHT = 150
MAX_WIDTH = 800
MAX_HEIGHT = 600
DPI = 300

# MongoDB connection
# client = MongoClient('localhost', 27017)
# db = client['image_recog']

labels = {
    0: "Aadhaar",
    1: "Canada ID",
    2: "Czeh ID",
    3: "Denmark PID",
    4: "Driving Licence",
    5: "Finland ID",
    6: "Israel NID",
    7: "PAN",
    8: "Passport",
    9: "Polan ID",
    10: "Singapore ID",
    11: "South Africa ID",
    12: "Spain SSN",
    13: "UK DL",
    14: "US Passport",
    15: "Utility",
    16: "Voter ID"
}

def resource_path(relative_path):
    """
    Get the absolute path to the resource.

    This function handles both development and PyInstaller bundle scenarios.

    Parameters:
        relative_path (str): The relative path to the resource.

    Returns:
        str: The absolute path to the resource.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model_path = resource_path('foreign_indian_classificiation_model.keras')

# Load the model
cnn_model = load_model(model_path)

def preprocess_image_classification(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """
    Preprocess an image for classification.

    This function converts the image to RGB if needed, resizes it, and normalizes it.

    Parameters:
        img (PIL.Image): The image to preprocess.
        img_width (int): The target width of the image.
        img_height (int): The target height of the image.

    Returns:
        np.array: The preprocessed image array.
    """
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]
    return img_array

def perform_ocr(img):
    """
    Perform Optical Character Recognition (OCR) on an image.

    This function uses Tesseract to extract text from the image.

    Parameters:
        img (PIL.Image): The image to perform OCR on.

    Returns:
        str: The extracted text from the image.
    """

    custom_config = '--psm 11 --oem 3'
    return pytesseract.image_to_string(img, lang='eng', config=custom_config)

def extract_info(text, doc_type):
    """
    Extract specific information from the text based on the document type.

    This function uses regular expressions to find and extract patterns such as PAN numbers or Aadhaar numbers.

    Parameters:
        text (str): The text to extract information from.
        doc_type (str): The type of document (e.g., 'PAN', 'Aadhaar').

    Returns:
        dict: A dictionary of extracted information.
    """
    patterns = {
        'PAN': {'PAN Number': r'([A-Z]{5}[0-9]{4}[A-Z]{1})'},
        'Aadhaar': {'Aadhaar Number': r'''(?<!\d)((?:[०-९]{4}[-\s]?[०-९]{4}[-\s]?[०-९]{4})|(?:[௦-௯]{4}[-\s]?[௦-௯]{4}[-\s]?[௦-௯]{4})|(?:[౦-౯]{4}[-\s]?[౦-౯]{4}[-\s]?[౦-౯]{4})|(?:[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}))(?!\d)'''},
        'Voter ID': {'Voter ID': r'\b[A-Za-z]{3}[0-9]{7}\b'},
        'Driving Licence': {'License Number': r'([A-Z]{2}\d{2} \d{11})'},
        'Passport': {'Passport Number': r'([A-Z]{1}[0-9]{7})'},
        'US Passport': {' US Passport Number': r'([0-9]{9})'},
        'Canada ID': {'Canada ID Number': r'(\d{4}-\d{4})'},
        'Czeh ID': {},
        'Denmark PID': {'Denmark ID Number': r'\d{6}-\d{4}'},
        'Finland ID': {'Finland ID Number': r'([0-9]{10})'},
        'Israel NID': {'Israel ID Number': r'([0-9]{9})'},
        'Polan ID': {'Poland ID Number': r'([0-9]{12})'},
        'Singapore ID': {'Singapore ID Number': r'[S][0-9]{7}[A-Z]'},
        'South Africa ID': {'South Africa ID Number': r'([0-9]{13})'},
        'Spain SSN': {'Spain ID Number': r"\b\d{11}\b"},
        'UK DL': {'Uk DL Number': r'[A-Z]{2}\d{6}[A-Z]{2}'},
    }.get(doc_type, {})

    extracted_info = {key: match.group(0) for key, pattern in patterns.items() if (match := re.search(pattern, text))}
    return extracted_info

def reclassify_utility(text):
    """
    Reclassify images labeled as 'Utility' based on specific keywords found in the text.

    This function searches for keywords associated with different document types to reclassify Utility images.

    Parameters:
        text (str): The text to search for keywords.

    Returns:
        str: The reclassified document type, or 'Utility' if no keywords are found.
    """

    keywords = {
        'Aadhaar': ['Aadhaar', 'UIDAI'],
        'Driving Licence': ['Driving Licence', 'DL No'],
        'PAN': ['Permanent Account Number', 'Income Tax Department'],
        'Passport': ['Passport', 'Republic of India'],
        'US Passport': ['Passport', 'United States of America'],
        'Voter ID': ['ELECTION', 'COMMISSION', 'ELECTION COMMISSION OF INDIA', 'IDENTITY CARD']
    }

    for doc_type, words in keywords.items():
        if any(re.search(word, text, re.IGNORECASE) for word in words):
            return doc_type
    return 'Utility'

def process_pdf_and_extract_images(pdf_path):
    """
    Process a PDF file and extract images from it.

    This function uses PyMuPDF to extract images from each page of the PDF.

    Parameters:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of extracted images as PIL.Image objects.
    """
    pdf_reader = fitz.open(pdf_path)
    images = []

    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = pdf_reader.extract_image(xref)
            img = Image.open(io.BytesIO(base_image["image"])).convert("RGBA")
            images.append(img)

    pdf_reader.close()
    return images

def resize_image(img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Resize an image while maintaining its aspect ratio.

    This function resizes the image so that it fits within the specified maximum width and height.

    Parameters:
        img (PIL.Image): The image to resize.
        max_width (int): The maximum width of the resized image.
        max_height (int): The maximum height of the resized image.

    Returns:
        PIL.Image: The resized image.
    """
    width, height = img.size
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        img = img.resize(new_size, Image.LANCZOS)
        img.info['dpi'] = (300, 300)
    return img

def process_single_image(img, file_path):
    """
    Process a single image for classification and information extraction.

    This function preprocesses the image, predicts its class using the CNN model, performs OCR to extract text, 
    and reclassifies Utility images based on the text content.

    Parameters:
        img (PIL.Image): The image to process.
    """

    img_array = preprocess_image_classification(img)
    predictions = cnn_model.predict(img_array)
    predicted_probability = np.max(predictions)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = labels[predicted_class[0]] if predicted_probability >= 0.5 else "Utility"

    img = resize_image(img)
    text = perform_ocr(img)

    if predicted_label == "Utility":
        new_label = reclassify_utility(text)
        if new_label != "Utility":
            predicted_label = new_label

    extracted_info = extract_info(text, predicted_label)

    return {
        "predicted_label": predicted_label,
        "probability": predicted_probability,
        "extracted_info": extracted_info
    }

# def save_to_db(file_path, predicted_label, extracted_info):
#     data = {
#         "file_path": file_path,
#         "predicted_label": predicted_label,
#         "extracted_info": extracted_info
#     }
#     db.abcd.insert_one(data)
#     print(f"Information from {file_path} has been saved to the database")

def main(file_path):
    """
    Main function to classify and extract information from an image or PDF.

    This function handles both image and PDF files, processes each image found, and uses threading for parallel processing.

    Parameters:
        file_path (str): The path to the image or PDF file.
    """
    start_time = time.time()

    if file_path.endswith('.pdf'):
        images = process_pdf_and_extract_images(file_path)
    else:
        images = [Image.open(file_path)]

    # Using ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_image, img, file_path) for img in images]
        for future in as_completed(futures):
            result = future.result()  # This will raise any exceptions encountered during processing

            # Custom formatted output
            print(f"Predicted Label: {result['predicted_label']}")
            print("Extracted Information:")
            for key, value in result['extracted_info'].items():
                print(f"  {key}: {value}")

            # Call save_to_db after processing each image
            # save_to_db(file_path, result['predicted_label'], result['extracted_info'])

    print(f"Time Taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    while True:
        file_path = input("Enter the image or PDF file path (or type 'exit' to quit): ")
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.lower() == 'exit':
            break
        main(file_path)


