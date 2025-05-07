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
tf.config.optimizer.set_jit(True)  # Enable XLA
import streamlit as st
###############################################################################################################################################################################################################################################################

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
###############################################################################################################################################################################################################################################################

# Constants
IMG_WIDTH = 150
IMG_HEIGHT = 150
MAX_WIDTH = 800
MAX_HEIGHT = 600
DPI = 300
###############################################################################################################################################################################################################################################################

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

###############################################################################################################################################################################################################################################################

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller bundle """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

###############################################################################################################################################################################################################################################################

model_path = resource_path('my_new_model.keras')
###############################################################################################################################################################################################################################################################

cnn_model = load_model(model_path)
###############################################################################################################################################################################################################################################################

def preprocess_image_classification(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """
    Preprocesses an image for input into a classification model.

    This function takes an image as input, converts it to RGB format if necessary, 
    resizes it to the specified dimensions, and then normalizes the pixel values to the range [0, 1].
    The resulting image is returned as a NumPy array suitable for input into a deep learning model.

    Args:
        img (PIL.Image.Image): The input image to be preprocessed.
        img_width (int, optional): The target width for resizing the image. Defaults to IMG_WIDTH.
        img_height (int, optional): The target height for resizing the image. Defaults to IMG_HEIGHT.

    Returns:
        numpy.ndarray: The preprocessed image as a 4D NumPy array with shape (1, img_height, img_width, 3).
                       The pixel values are normalized to the range [0, 1].
    """
    # Convert image to RGB if it has an alpha channel (RGBA)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':  # If the image is in any other mode, convert it to RGB
        img = img.convert('RGB')
        
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]
    return img_array

###############################################################################################################################################################################################################################################################

# Function to perform OCR on the image
def perform_ocr(img):
    """
    Performs Optical Character Recognition (OCR) on the given image.

    This function uses Tesseract OCR to extract text from the provided image. 
    The OCR operation is configured to use page segmentation mode 11 (sparse text) 
    and OCR engine mode 3 (default, based on what is available).

    Args:
        img (PIL.Image.Image): The input image on which OCR is to be performed.

    Returns:
        str: The text extracted from the image.
    """
    custom_config = '--psm 11 --oem 3'
    return pytesseract.image_to_string(img, lang='eng', config=custom_config)
###############################################################################################################################################################################################################################################################

# Function to extract information from a document
def extract_info(text, doc_type):
    """
    Extracts specific information from a given text based on the type of document.

    This function uses regular expressions to search for and extract key details 
    (like PAN numbers, Aadhaar numbers, Voter IDs, etc.) from the input text based 
    on the specified document type. The extracted information is returned as a dictionary.

    Args:
        text (str): The input text from which information is to be extracted.
        doc_type (str): The type of document from which to extract information. 
                        Supported types include 'PAN', 'Aadhaar', 'Voter ID', 'Driving Licence', 
                        'Passport', and 'US Passport'.

    Returns:
        dict: A dictionary containing the extracted information, where keys are 
              the types of information (e.g., 'PAN Number', 'Aadhaar Number') 
              and values are the corresponding extracted values.

    Example:
        extracted_info = extract_info("Your PAN is ABCDE1234F", "PAN")
        # Output: {'PAN Number': 'ABCDE1234F'}
    """
    patterns = {
        'PAN': {'PAN Number': r'([A-Z]{5}[0-9]{4}[A-Z]{1})'},
        'Aadhaar': {'Aadhaar Number': r'''(?<!\d)((?:[०-९]{4}[-\s]?[०-९]{4}[-\s]?[०-९]{4})|(?:[௦-௯]{4}[-\s]?[௦-௯]{4}[-\s]?[௦-௯]{4})|(?:[౦-౯]{4}[-\s]?[౦-౯]{4}[-\s]?[౦-౯]{4})|(?:[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}))(?!\d)'''},
        'Voter ID': {'Voter ID': r'\b[A-Za-z]{3}[0-9]{7}\b'},
        'Driving Licence': {'License Number': r'([A-Z]{2}\d{2} \d{11})'},
        'Passport': {'Passport Number': r'([A-Z]{1}[0-9]{7})'},
        'US Passport': {'Passport Number': r'([0-9]{9})'},
    }.get(doc_type, {})

    extracted_info = {key: match.group(0) for key, pattern in patterns.items() if (match := re.search(pattern, text))}
    for key, value in extracted_info.items():
        print(f"{key}: {value}")
    
    return extracted_info
###############################################################################################################################################################################################################################################################

# Function to check for keywords and reclassify Utility images
def reclassify_utility(text):
    """
    Classifies a document based on the presence of specific keywords in the text.

    This function analyzes the input text to determine the type of document it belongs to 
    by searching for predefined keywords associated with different document types 
    (e.g., Aadhaar, Driving Licence, PAN, Passport, US Passport, Voter ID). 
    If a match is found, the corresponding document type is returned; otherwise, 
    the text is classified as 'Utility'.

    Args:
        text (str): The input text to be analyzed for document classification.

    Returns:
        str: The type of document determined based on the keywords found in the text.
             Returns 'Utility' if no specific document type keywords are found.

    Example:
        doc_type = reclassify_utility("This is a Passport from the Republic of India.")
        # Output: 'Passport'
    """
    
    keywords = {
        'Aadhaar': ['Aadhaar', 'UIDAI'],
        'Driving Licence': ['Driving Licence', 'DL No'],
        'PAN': ['Permanent Account Number', 'Income Tax Department'],
        'Passport': ['Passport', 'Republic of India'],
        'US Passport': ['Passport', 'United States of America'],
        'Voter ID': ['ELECTION','COMMISSION','ELECTION COMMISSION OF INDIA','IDENTITY CARD']
    }

    for doc_type, words in keywords.items():
        if any(re.search(word, text, re.IGNORECASE) for word in words):
            print(f"Keyword found for document type '{doc_type}'")
            return doc_type
    return 'Utility'
###############################################################################################################################################################################################################################################################

# Function to process PDF and extract images
def process_pdf_and_extract_images(pdf_path):
    """
    Extracts images from each page of a PDF file.

    This function processes a PDF file by reading each page and extracting all images embedded within it. 
    The extracted images are converted to RGBA format and returned as a list of PIL Image objects.

    Args:
        pdf_path (str): The file path to the PDF document from which images will be extracted.

    Returns:
        list: A list of PIL.Image.Image objects representing the extracted images from the PDF.
              Each image is in RGBA format.

    Example:
        images = process_pdf_and_extract_images("sample.pdf")
        # Output: [<PIL.Image.Image image mode=RGBA size=...>, <PIL.Image.Image image mode=RGBA size=...>]
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
###############################################################################################################################################################################################################################################################

# Function to resize images maintaining aspect ratio
def resize_image(img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Resizes an image to fit within specified maximum dimensions while maintaining the aspect ratio.

    This function takes an image and resizes it if its dimensions exceed the specified 
    maximum width or height. The resizing is done proportionally to ensure that the image 
    maintains its original aspect ratio. The resized image is returned with updated DPI information.

    Args:
        img (PIL.Image.Image): The input image to be resized.
        max_width (int, optional): The maximum allowed width for the resized image. 
                                   Defaults to MAX_WIDTH.
        max_height (int, optional): The maximum allowed height for the resized image. 
                                    Defaults to MAX_HEIGHT.

    Returns:
        PIL.Image.Image: The resized image, maintaining the aspect ratio, with updated DPI information.

    Example:
        resized_img = resize_image(original_img, max_width=800, max_height=600)
        # Output: <PIL.Image.Image image mode=... size=...>
    """
    width, height = img.size
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        img = img.resize(new_size, Image.LANCZOS)
        img.info['dpi'] = (DPI, DPI)  
    return img
###############################################################################################################################################################################################################################################################

# Function to process a single image (for use in threading)
def process_single_image(img):
    """
    Processes a single image to classify its content, perform OCR, and extract relevant information.

    This function first preprocesses the input image for classification using a CNN model. 
    It then predicts the class label with an associated probability. If the predicted probability 
    is less than 0.5, the image is classified as "Utility." The function resizes the image and 
    performs OCR to extract text. If the image is classified as "Utility," it attempts to reclassify 
    based on keywords found in the extracted text. Finally, it extracts and prints relevant information 
    based on the determined label.

    Args:
        img (PIL.Image.Image): The input image to be processed.

    Returns:
        None

    Example:
        process_single_image(image)
        # Output: Predicted Label: Passport, Probability: 0.85
        #         Reclassified Label: None
        #         Extracted Info: {'Passport Number': 'A1234567'}
    """
    img_array = preprocess_image_classification(img)
    predictions = cnn_model.predict(img_array)
    predicted_probability = np.max(predictions)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = labels[predicted_class[0]] if predicted_probability >= 0.5 else "Utility"
    print(f"Predicted Label: {predicted_label}, Probability: {predicted_probability}")

    img = resize_image(img)
    text = perform_ocr(img)

    if predicted_label == "Utility":
        new_label = reclassify_utility(text)
        if new_label != "Utility":
            predicted_label = new_label
            print(f"Reclassified Label: {predicted_label}")

    extract_info(text, predicted_label)
###############################################################################################################################################################################################################################################################

# Main function to classify and extract information from the image or PDF
def main(file_path):
    start_time = time.time()

    if file_path.endswith('.pdf'):
        images = process_pdf_and_extract_images(file_path)
    else:
        images = [Image.open(file_path)]

    # Using ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_image, img) for img in images]
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions encountered during processing

    print(f"Time Taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    while True:
        file_path = input("Enter the image or PDF file path (or type 'exit' to quit): ")
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.lower() == 'exit':
            break
        main(file_path)

###############################################################################################################################################################################################################################################################
