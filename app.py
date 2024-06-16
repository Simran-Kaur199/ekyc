import cv2
import numpy as np
import os
import logging
import streamlit as st
import mysql.connector
from sqlalchemy import text
from preprocess import read_image, extract_id_card, save_image
from ocr_engine import extract_text
from postprocess import extract_information
from face_verification import detect_and_extract_face, face_comparison, get_face_embeddings
from msql_op import insert_records, fetch_records, check_duplicacy

# {
#   "ID": "CCNPA",
#   "Name": "BIBEK RAUTH",
#   "Father's Name": "AJAY RAUTH",
#   "DOB": "14/09/1994",
#   "ID Type": "PAN"
# }


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

# Set wider page layout
def wider_page():
    max_width_str = "max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Page layout set to wider configuration.")

# Customized Streamlit theme
def set_custom_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6; /* Set background color */
                color: #333333; /* Set text color */
            }
            .sidebar .sidebar-content {
                background-color: #ffffff; /* Set sidebar background color */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Custom theme applied to Streamlit app.")

# Sidebar
def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    option = st.sidebar.selectbox("Select ID Card", ("PAN", "Aadhar"), label_visibility="collapsed")
    logging.info(f"ID card type selected: {option}")
    return option

# Header
def header_section(option):
    if option == "Aadhar":
        st.title("Registration Using Aadhar Card")
        logging.info("Header set for Aadhar Card registration.")
    elif option == "PAN":
        st.title("Registration Using PAN Card")
        logging.info("Header set for PAN Card registration.")

# Main content
# def main_content(image_file, face_image_file, conn):
#     if image_file is not None:
#         face_image = read_image(face_image_file, is_uploaded=True)
#         logging.info("Face image loaded.")
#         if face_image is not None:
#             image = read_image(image_file, is_uploaded=True)
#             logging.info("ID card image loaded.")
#             image_roi, _ = extract_id_card(image)
#             logging.info("ID card ROI extracted.")
#             face_image_path2 = detect_and_extract_face(img=image_roi)
#             face_image_path1 = save_image(face_image, "face_image.jpg", path="data\\02_intermediate_data")
#             logging.info("Faces extracted and saved.")
#             is_face_verified = face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)
#             logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")

#             if is_face_verified:
#                 extracted_text = extract_text(image_roi)
#                 text_info = extract_information(extracted_text)
#                 logging.info("Text extracted and information parsed from ID card.")
#                 records = fetch_records(text_info)
#                 if records.shape[0] > 0:
#                     st.write(records.shape)
#                     st.write(records)
#                 is_duplicate = check_duplicacy(text_info)
#                 if is_duplicate:
#                     st.write(f"User already present with ID {text_info['ID']}")
#                 else: 
#                     st.write(text_info)
#                     text_info['DOB'] = text_info['DOB'].strftime('%Y-%m-%d')
#                     text_info['Embedding'] =  get_face_embeddings(face_image_path1)
#                     insert_records(text_info)
#                     logging.info(f"New user record inserted: {text_info['ID']}")
                    
#             else:
#                 st.error("Face verification failed. Please try again.")

#         else:
#             st.error("Face image not uploaded. Please upload a face image.")
#             logging.error("No face image uploaded.")

#     else:
#         st.warning("Please upload an ID card image.")
#         logging.warning("No ID card image uploaded.")

# def main_content(image_file, face_image_file, mycursor, mydb):
#     if image_file is not None:
#         try:
#             # Read the uploaded ID card image
#             image_bytes = np.frombuffer(image_file.read(), np.uint8)
#             image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
#             if image is None:
#                 st.error("Error reading ID card image. Please upload a valid image.")
#                 logging.error("Error reading ID card image. The image is None.")
#                 return
#             logging.info("ID card image loaded.")

#             # Read the uploaded face image
#             if face_image_file is not None:
#                 face_image_bytes = np.frombuffer(face_image_file.read(), np.uint8)
#                 face_image = cv2.imdecode(face_image_bytes, cv2.IMREAD_COLOR)
#                 if face_image is None:
#                     st.error("Error reading face image. Please upload a valid image.")
#                     logging.error("Error reading face image. The image is None.")
#                     return
#                 logging.info("Face image loaded.")

#                 image_roi, _ = extract_id_card(image)
#                 logging.info("ID card ROI extracted.")
#                 face_image_path2 = detect_and_extract_face(img=image_roi)
#                 face_image_path1 = save_image(face_image, "face_image.jpg", path="data\\02_intermediate_data")
#                 logging.info("Faces extracted and saved.")
#                 is_face_verified = face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)
#                 logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")

#                 if is_face_verified:
#                     extracted_text = extract_text(image_roi)
#                     text_info = extract_information(extracted_text)
#                     logging.info("Text extracted and information parsed from ID card.")
#                     records = fetch_records(text_info, mycursor)
#                     if not records.empty:
#                         st.write(records.shape)
#                         st.write(records)
#                     is_duplicate = check_duplicacy(text_info, mycursor)
#                     if is_duplicate:
#                         st.write(f"User already present with ID {text_info['ID']}")
#                     else:
#                         st.write(text_info)
#                         text_info['DOB'] = text_info['DOB'].strftime('%Y-%m-%d')
#                         text_info['Embedding'] = get_face_embeddings(face_image_path1)
#                         insert_records(text_info, mycursor, mydb)
#                         logging.info(f"New user record inserted: {text_info['ID']}")

#                 else:
#                     st.error("Face verification failed. Please try again.")

#             else:
#                 st.error("Face image not uploaded. Please upload a face image.")
#                 logging.error("No face image uploaded.")

#         except Exception as e:
#             st.error(f"Error reading image: {e}")
#             logging.error(f"Error reading image: {e}")

#     else:
#         st.warning("Please upload an ID card image.")
#         logging.warning("No ID card image uploaded.")

def main_content(image_file, face_image_file, mycursor, mydb):
    if image_file is not None:
        face_image = read_image(face_image_file, is_uploaded=True)
        logging.info("Face image loaded.")
        if face_image is not None:
            image = read_image(image_file, is_uploaded=True)
            logging.info("ID card image loaded.")
            image_roi, _ = extract_id_card(image)
            logging.info("ID card ROI extracted.")
            
            # Detect and extract face from the ID card
            face_image_path2 = detect_and_extract_face(img=image_roi)
            if face_image_path2 is None:
                st.error("Failed to detect and extract face from ID card.")
                logging.error("Failed to detect and extract face from ID card.")
                return
            
            # Save the uploaded face image
            face_image_path1 = save_image(face_image, "face_image.jpg", path="data\\02_intermediate_data")
            if face_image_path1 is None:
                st.error("Failed to save face image.")
                logging.error("Failed to save face image.")
                return
            
            logging.info("Faces extracted and saved.")
            
            # Verify the faces
            is_face_verified = face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)
            logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")

            if is_face_verified:
                # Extract text from ID card and process it
                extracted_text = extract_text(image_roi)
                text_info = extract_information(extracted_text)
                logging.info("Text extracted and information parsed from ID card.")
                
                # Fetch records from the database
                records = fetch_records(text_info, mycursor)
                if not records.empty:
                    st.write(records.shape)
                    st.write(records)
                    
                # Check for duplicacy in the database
                is_duplicate = check_duplicacy(text_info, mycursor)
                if is_duplicate:
                    st.write(f"User already present with ID {text_info['ID']}")
                else:
                    st.write(text_info)
                    text_info['DOB'] = text_info['DOB'].strftime('%Y-%m-%d')
                    text_info['Embedding'] = get_face_embeddings(face_image_path1)
                    insert_records(text_info, mycursor, mydb)
                    logging.info(f"New user record inserted: {text_info['ID']}")
            else:
                st.error("Face verification failed. Please try again.")
        else:
            st.error("Face image not uploaded. Please upload a face image.")
            logging.error("No face image uploaded.")
    else:
        st.warning("Please upload an ID card image.")
        logging.warning("No ID card image uploaded.")
        
def main():
    # Load database credentials from secrets.toml
    db_config = st.secrets["mysql"]
    
    # Establish a connection to MySQL Server
    mydb = mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"]
    )
    mycursor = mydb.cursor()
    print("Connection Established")
    logging.info("MySQL database connection established.")

    wider_page()
    set_custom_theme()
    option = sidebar_section()
    header_section(option)
    image_file = st.file_uploader("Upload ID Card")
    if image_file is not None:
        face_image_file = st.file_uploader("Upload Face Image")
        main_content(image_file, face_image_file, mycursor, mydb)

if __name__ == "__main__":
    main()