import patterns  
import validation
import keywords

def tag_pii_context(text):
    result = {
        "AADHAR_CARD": {},
        "PAN_CARD": {},
        "Mobile": {},
        "Driving_licence": {},
        "Credit_Card": {},
        "Debit_Card": {},
        "ip_address": {},
        "mac_address": {},
    }

    aadhaar_numbers = patterns.aadhaar_pattern.findall(text)
    # print("aadhaar_numbers: ",aadhaar_numbers)
    for i, num in enumerate(aadhaar_numbers):
        if num in text:
            start_index = text.index(num)
            context = start_index-50
            if context < 0:
                context = 0
            adhar_keyword = [item.lower() for item in keywords.adhar_keyword]
            if any(word in text[context:start_index].lower() for word in adhar_keyword):
                if validation.validate_aadhaar(num) :
                    result["AADHAR_CARD"][f"<AADHAR_CARD_{i+1}>"] = num

    pan_numbers = patterns.pan_pattern.findall(text)
    for i, num in enumerate(pan_numbers):
        if num in text:
            start_index = text.index(num)
            context = start_index-50
            if context < 0:
                context = 0
            context1 = text[context:start_index]
            pan_numbers_keywords = [item.lower() for item in keywords.pan_numbers_keywords]
            if any(word in context1.lower() for word in pan_numbers_keywords):
                if num[3] in ('t','f','p','c','h'):
                    result["PAN_CARD"][f"<PAN_CARD_{i+1}>"] = num

    mobile_numbers = patterns.mobile_number_pattern.findall(text)
    cleaned_items = [item.strip() for item in mobile_numbers]
    for i, num in enumerate(cleaned_items):
        if num in text:
            start_index = text.index(num)
            context = start_index-20
            if context < 0:
                context = 0
            context1 = text[context:start_index]
            phone_keyword = [item.lower() for item in keywords.phone_keyword]
            if any(word in context1.lower() for word in phone_keyword):
                is_valid, carrier_info, geo_info = validation.validate_number(num)
                if is_valid:
                    result["Mobile"][f"<Mobile_{i+1}>"] = num

    driving_licence_number  = patterns.driving_licence_pattern.findall(text)
    for i ,num in enumerate(driving_licence_number):
        if num in text:
            start_index = text.index(num)
            context = start_index -50
            if context < 0:
                context = 0
            context1 = text[context:start_index].lower()
            dl_keyword = [item.lower() for item in keywords.dl_keyword]
            if any(word in context1 for word in dl_keyword):
                result["Driving_licence"][f"Driving_licence_{i+1}>"] = num

    visa_numbers = patterns.visa_pattern.findall(text)
    for i ,num in enumerate(visa_numbers):
        if num in text:
            start_index = text.index(num)
            context = start_index -50
            if context < 0:
                context = 0

            context1 = text[context:start_index].lower()
            cc_keyword = [item.lower() for item in keywords.cc_keyword]
            if any(word in context1 for word in cc_keyword):
                if validation.checkLuhn(num):
                    result["Credit_Card"][f"Credit_Card_{i+1}>"] = num

    debit_number = patterns.debit_card_pattern.findall(text)
    for i ,num in enumerate(debit_number):
            if num in text:
                start_index = text.index(num)
                context = start_index -50
                if context < 0:
                    context = 0
                context1 = text[context:start_index].lower()
                cc_keyword = [item.lower() for item in keywords.dbt_keyword]
                if any(word in context1 for word in cc_keyword):
                    if validation.checkLuhn(num):
                        result["Debit_Card"][f"Debit_Card_{i+1}>"] = num

    ip_address_num = patterns.ip_address_pattern.findall(text)
    for i , num in enumerate(ip_address_num):
        if num in text:
            start_index = text.index(num)
            context = start_index - 50
            if context <0:
                context = 0
            context1 = text[context:start_index].lower()
            ip_address_num_keywords = [item.lower() for item in keywords.ip_address_num_keywords]
            if any(word in context1 for word in ip_address_num_keywords):
                result["ip_address"][f"ip_address_{i+1}>"] = num

    # mac_address_num = patterns.mac_address_pattern.findall(text)
    # print("mac_address_num: ",mac_address_num)

    # for i , num in enumerate(mac_address_num):
    #     if num in text:
    #         start_index = text.index(num)
           
    #         context = start_index - 50
    #         if context <0:
    #             context = 0
    #         context1 = text[context:start_index].lower()
    #         mac_address_num_keywords = ["mac address","mac"]
    #         mac_address_num_keywords = [item.lower() for item in mac_address_num_keywords]
    #         if any(word in context1 for word in mac_address_num_keywords):
    #             result["mac_address"][f"mac_address_{i+1}>"] = num
    filtered_data = {key: value for key, value in result.items() if value}
    return filtered_data




# text = """debit card is 5105 1051 0510 5100,my friends contact number is 9860519512,My Credit Card number is 6011 1111 1111 1117,aadhaar card number is 237658701719 
# and my friends contact number is 9860519512 ,pan number is GQPPK8007C ,vechile registration number is GJ18 20090538447,my ip address is 192.168.1.1 and mac is 01:23:45:67:89:AB"""

# # text = """मेरा डेबिट कार्ड नंबर ५१०५ १०५१ ०५१० ५१०० है। मेरे दोस्त का संपर्क नंबर ९८६०५१९५१२ है। मेरा क्रेडिट कार्ड नंबर ४१११ ११११ ११११ ११११ है और मेरा आधार कार्ड नंबर २३७६५८७०१७१९ है। 
# # मेरे पैन नंबर GQPPK८००७C है। मेरा वाहन पंजीकरण नंबर GJ18 20090538447 है। 
# # मेरा आईपी पता 192.168.1.1 है और मेरा मैक पता १९२.१६८.१.१:एबी है। मेरे दोस्त का संपर्क नंबर फिर से ९८६०५१९५१२ है।"""

# print("tag_pii:", tag_pii_context(text))