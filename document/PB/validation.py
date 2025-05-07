import phonenumbers
from phonenumbers import carrier, geocoder, is_valid_number

multiplication_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

permutation_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]
inverse_table = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

def inv_array(array):
    if isinstance(array, (int, str)):
        array = list(map(int, str(array)))
    return array[::-1]

def generate(array):
    c = 0
    inverted_array = inv_array(array)
    for i in range(len(inverted_array)):
        c = multiplication_table[c][permutation_table[(i + 1) % 8][inverted_array[i]]]
    return inverse_table[c]

def validate(array):
    c = 0
    inverted_array = inv_array(array)
    for i in range(len(inverted_array)):
        c = multiplication_table[c][permutation_table[i % 8][inverted_array[i]]]
    return c == 0

def validate_aadhaar(aadhaar_string):
    aadhaar_string=aadhaar_string.replace(" ", "")
    if len(aadhaar_string) != 12:
        raise ValueError('Aadhaar numbers should be 12 digit in length')
    if not aadhaar_string.isdigit():
        raise ValueError('Aadhaar numbers must contain only numbers')
    aadhaar_array = list(map(int, aadhaar_string))
    to_check_checksum = aadhaar_array.pop()
    return generate(aadhaar_array) == to_check_checksum

def validate_number(number, region="IN"):
    try:
        # Parse the number with the given region
        parsed_number = phonenumbers.parse(number, region)

        # Check if the number is valid
        if is_valid_number(parsed_number):
            # Retrieve carrier information
            carrier_info = carrier.name_for_number(parsed_number, "en")

            # Retrieve geographic information
            geo_info = geocoder.description_for_number(parsed_number, "en")

            return True, carrier_info, geo_info
        else:
            return False, None, None

    except phonenumbers.NumberParseException:
        return False, None, None

def checkLuhn(cardNo):
    # Remove spaces and hyphens
    cardNo = cardNo.replace(' ', '').replace('-', '')

    # Convert Devanagari digits to ASCII digits
    cardNo = ''.join(str(ord(c) - ord('०')) if '०' <= c <= '९' else c for c in cardNo)

    # Valid lengths for debit and credit card numbers
    valid_lengths = {12, 13, 14, 15, 16, 17, 18, 19}

    # Check for valid length
    if len(cardNo) not in valid_lengths:
        return False

    nSum = 0
    isSecond = False

    # Process each digit from right to left
    for i in range(len(cardNo) - 1, -1, -1):
        d = ord(cardNo[i]) - ord('0')  # Convert character to digit

        if isSecond:
            d = d * 2  # Double every second digit
            if d > 9:  # If result is greater than 9, subtract 9
                d -= 9

        nSum += d
        isSecond = not isSecond

    # Check if the sum is a multiple of 10
    return nSum % 10 == 0