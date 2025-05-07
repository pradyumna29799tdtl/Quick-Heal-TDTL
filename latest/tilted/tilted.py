import cv2
import numpy as np
import pytesseract
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pytesseract import Output

def checkLuhn(cardNo):
        cardNo = cardNo.replace(' ', '').replace('-', '')
        cardNo = ''.join(str(ord(c) - ord('०')) if '०' <= c <= '९' else c for c in cardNo)  # Convert Devanagari digits to ASCII digits
        if len(cardNo) != 16:
            return False
        nSum = 0
        isSecond = False
        for i in range(len(cardNo) - 1, -1, -1):
            d = ord(cardNo[i]) - ord('0')
            if isSecond:
                d = d * 2
            nSum += d // 10
            nSum += d % 10
            isSecond = not isSecond
        return nSum % 10 == 0

def devanagari_to_ascii(number):
    return ''.join(
        str(ord(char) - ord('०')) if '०' <= char <= '९' else char for char in number
    )

def tag_pii(text):
    result = {
        "AADHAR_CARD": {},
        "PAN_CARD": {},
        "DOB": {},
        "Voter_ID": {},
        "Mobile": {},
        "Email": {},
        "Landline": {},
        "Credit_Card": {},
        "mac_address":{},
        "Passport":{},
        "driving_license":{},
    }
    aadhaar_pattern = re.compile(r'(?<!\d)(\d{4}[-\s]?\d{4}[-\s]?\d{4})(?!\d)')
    driving_license = re.compile(r'\b[A-Z]{2}\s?\d{2}\s?\d{4}\s?\d{7}\b')
    pan_pattern = re.compile(r'[A-Z]{5}\d{4}[A-Z]?')
    voter_id_pattern = re.compile(r'[A-Z]{3}[0-9]{7}')
    mobile_number_pattern = re.compile(
        r'(?:(?:\+91|91|0)?[\s-]?[789]\d{9})'
        r'|(?:\+?\d{1,3}[\s-]?)?[789]\d{9}'
        r'|(?:[\u0966-\u096F\u09E6-\u09EF\u0AE6-\u0AEF\u0A66-\u0A6F\u0CE6-\u0CEF\u0D66-\u0D6F\u0BE6-\u0BEF\u0C66-\u0C6F\u0B66-\u0B6F]{10})'
        r'|\+९१\s\d{5}\s\d{5}'
        r'|\+?\d{2,3}\s?\d{10}'
        r'|\(\d{3}\)\s?\d{3}-\d{4}'
    )
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    passport_pattern = re.compile(r'\b[A-Z]{1}[0-9]{7}\b')
    DOB_pattern = re.compile(
        r'\b(?:'
        r'(?:(?:[0-3\u0966-\u0969][0-9\u0966-\u096F])[-/.](?:[0-1\u0966][0-9\u0966-\u096F])[-/.](?:[12\u0967-\u0968][09\u0966\u096F][0-9\u0966-\u096F]{2}))|'  # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        r'(?:(?:[0-1\u0966][0-9\u0966-\u096F])[-/.](?:[0-3\u0966-\u0969][0-9\u0966-\u096F])[-/.](?:[12\u0967-\u0968][09\u0966\u096F][0-9\u0966-\u096F]{2}))|'  # MM-DD-YYYY, MM/DD/YYYY, MM.DD.YYYY
        r'(?:(?:[12\u0967-\u0968][09\u0966\u096F][0-9\u0966-\u096F]{2})[-/.](?:[0-1\u0966][0-9\u0966-\u096F])[-/.](?:[0-3\u0966-\u0969][0-9\u0966-\u096F]))'   # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        r')\b'
    )
    # landline_number_pattern = re.compile(
    # r'''(?:(?:\+\d{1,2}\s?)?(?:\(\d{2,}\)|\d{2,})[-.\s]?(?:(?:\d{1,2}[-.\s]?){3,5}\d|\(\d{4}\)\s?\d{3}\s?\d{3}\s?\d))|(?:
    #     (?:[\u0966-\u096F\u09E6-\u09EF\u0AE6-\u0AEF\u0A66-\u0A6F\u0CE6-\u0CEF\u0D66-\u0D6F\u0BE6-\u0BEF\u0C66-\u0C6F\u0B66-\u0B6F]{6,8})
    # )
    # ''',
    # re.VERBOSE
    # )
    card_pattern = re.compile(
        r'\b(?:[\d\u0966-\u096F\u09E6-\u09EF\u0AE6-\u0AEF\u0A66-\u0A6F\u0CE6-\u0CEF\u0D66-\u0D6F\u0BE6-\u0BEF\u0C66-\u0C6F\u0B66-\u0B6F\u1C50-\u1C59]{4}[-\s]?){3}'
        r'[\d\u0966-\u096F\u09E6-\u09EF\u0AE6-\u0AEF\u0A66-\u0A6F\u0CE6-\u0CEF\u0D66-\u0D6F\u0BE6-\u0BEF\u0C66-\u0C6F\u0B66-\u0B6F\u1C50-\u1C59]{4}\b'
    )
    mac_address_pattern = re.compile(
        r'(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})'
        r'|(?:[0-9A-Fa-f]{4}\.){2}(?:[0-9A-Fa-f]{4})'
        r'|(?:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]{2}\.[0-9A-Fa-f]{2}\.[0-9A-Fa-f]{2}\.[0-9A-Fa-f]{2}\.[0-9A-Fa-f]{2})'
        r'|(?:[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2})'
    )
    aadhaar_numbers = aadhaar_pattern.findall(text)
    # valid_aadhaar_numbers = [num for num in aadhaar_numbers if is_valid_aadhaar_number(num.replace(' ', '').replace('-', ''))]
    for i, num in enumerate(aadhaar_numbers):
        result["AADHAR_CARD"][f"<AADHAR_CARD_{i+1}>"] = num
    pan_numbers = pan_pattern.findall(text)
    for i, num in enumerate(pan_numbers):
        result["PAN_CARD"][f"<PAN_CARD_{i+1}>"] = num
    voter_id_numbers = voter_id_pattern.findall(text)
    for i, num in enumerate(voter_id_numbers):
        result["Voter_ID"][f"<Voter_ID_{i+1}>"] = num
    mobile_numbers = mobile_number_pattern.findall(text)
    for i, num in enumerate(mobile_numbers):
        result["Mobile"][f"<Mobile_{i+1}>"] = num
    email_addresses = email_pattern.findall(text)
    for i, email in enumerate(email_addresses):
        result["Email"][f"<Email_{i+1}>"] = email
    mac_address_patterns = mac_address_pattern.findall(text)
    for i, num in enumerate(mac_address_patterns):
        result["mac_address"][f"<mac_address_{i+1}>"] = num
    driving_license = driving_license.findall(text)
    for i, num in enumerate(driving_license):
        result["driving_license"][f"<driving_license{i+1}>"] = num
    #landline_numbers = landline_number_pattern.findall(text)
    # for i, num in enumerate(landline_numbers):
    #     result["Landline"][f"<Landline_{i+1}>"] = num
    DOB_patterns = DOB_pattern.findall(text)
    for i, num in enumerate(DOB_patterns):
        ascii_dob = devanagari_to_ascii(num)
        result["DOB"][f"<DOB_pattern_{i+1}>"] = ascii_dob
    card_numbers = card_pattern.findall(text)
    valid_card_numbers = []
    for card in card_numbers:
        if checkLuhn(card):
            valid_card_numbers.append(card)
    for i, card in enumerate(valid_card_numbers):
        result["Credit_Card"][f"<Credit_Card_{i+1}>"] = card
    passport_number = passport_pattern.findall(text)
    for i, card in enumerate(passport_number):
        result["Passport"][f"<Passport_{i+1}>"] = card
    filtered_data = {key: value for key, value in result.items() if value}
    return filtered_data

def deskew(image):
     coords = np.column_stack(np.where(image > 0))
     angle = cv2.minAreaRect(coords)[-1]
     print(angle)
     if angle < -45:
         angle = -(90 + angle)
     else:
         angle = -angle
     (h, w) = image.shape[:2]
     center = (w // 2, h // 2)
     M = cv2.getRotationMatrix2D(center, angle, 1.0)
     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
      
     return rotated

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


# temp_image_path = r"D:\PradyumnaBadave\QuickHeal\ocr\Clockwise\img-80.png"
# img = cv2.imread(temp_image_path)
# if img is None:
#     print({'error': 'Failed to read image file'})
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kernel = np.ones((2, 2), np.uint8)
# opening_image = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
# edged_image = cv2.Canny(gray_img, 75, 200)
# deskewed_image = deskew(gray_img)

# coords = np.column_stack(np.where(edged_image > 0))
# angle = cv2.minAreaRect(coords)[-1]
# contours, _ = cv2.findContours(opening_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)
# doc_cnts = None
# for contour in contours:
#     peri = cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
#     if len(approx) == 4:
#         doc_cnts = approx
#         break
# if doc_cnts is None:
#     print({'error': 'Could not find document contours'}, status=400)
# warped = four_point_transform(img, doc_cnts.reshape(4, 2))

# warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# plt.imshow(warped_gray)
# plt.axis('off')  # Turn off axis labels
# plt.show()

# osd = pytesseract.image_to_osd(warped_gray, output_type=Output.DICT)
# print(f"Rotation: {osd['rotate']}")
# print(f"Orientation: {osd['orientation']}")
# print(f"Script: {osd['script']}")

# # Calculate the center of the image
# (h, w) = warped_gray.shape[:2]
# center = (w // 2, h // 2)

# # Define the rotation matrix
# M = cv2.getRotationMatrix2D(center, osd['rotate'] if osd['rotate'] == 0 else -1*osd['rotate'], 1.0)

# # Rotate the image
# rotated_img = cv2.warpAffine(warped_gray, M, (w, h))

# plt.imshow(rotated_img)
# plt.axis('off')  # Turn off axis labels
# plt.show()

# extracted_text = pytesseract.image_to_string(rotated_img, lang='eng')
# tag_dict = tag_pii(extracted_text)
# print({'extracted_text': extracted_text,'label': 'Adhar','tags_content': tag_dict})

def process_image(temp_image_path):
    img = cv2.imread(temp_image_path)
    if img is None:
        return {'error': 'Failed to read image file'}

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    opening_image = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    edged_image = cv2.Canny(gray_img, 75, 200)
    
    coords = np.column_stack(np.where(edged_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    contours, _ = cv2.findContours(opening_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_cnts = None
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        if len(approx) == 4:
            doc_cnts = approx
            break
    
    if doc_cnts is None:
        return {'error': 'Could not find document contours', 'status': 400}
    
    warped = four_point_transform(img, doc_cnts.reshape(4, 2))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(warped_gray)
    plt.axis('off')
    plt.show()
    
    osd = pytesseract.image_to_osd(warped_gray, output_type=Output.DICT)
    print(f"Rotation: {osd['rotate']}")
    print(f"Orientation: {osd['orientation']}")
    print(f"Script: {osd['script']}")
    
    (h, w) = warped_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, osd['rotate'] if osd['rotate'] == 0 else -1*osd['rotate'], 1.0)
    rotated_img = cv2.warpAffine(warped_gray, M, (w, h))
    
    plt.imshow(rotated_img)
    plt.axis('off')
    plt.show()
    
    extracted_text = pytesseract.image_to_string(rotated_img, lang='eng')
    tag_dict = tag_pii(extracted_text)
    
    return {'label': 'Adhar', 'tags_content': tag_dict}


# You can call the function like this:

result = process_image(r"D:\PradyumnaBadave\QuickHeal\ocr\Clockwise\img-80.png")
print(result)