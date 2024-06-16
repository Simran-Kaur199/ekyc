import cv2
import numpy as np
import os
import logging
from utils import read_yaml, file_exists

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

config_path = "config.yaml"
config = read_yaml(config_path)

artifacts = config['artifacts']
intermediate_dir_path = artifacts['INTERMEDIATE_DIR']
contour_filename = artifacts['CONTOUR_FILE']

# def read_image(image_path, is_uploaded=False):
#     if is_uploaded:
#         try:
#             image_bytes = image_path.read()
#             img = cv2.imread(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
#             if img is None:
#                 logging.info("Failed to read image: {}".format(image_path))
#                 raise Exception("Failed to read image: {}".format(image_path))
#             return img
#         except Exception as e:
#             logging.info(f"Error reading image: {e}")
#             print("Error reading image:", e)
#             return None
#     else:
#         try:
#             img = cv2.imread(image_path)
#             if img is None:
#                 logging.info("Failed to read image: {}".format(image_path))
#                 raise Exception("Failed to read image: {}".format(image_path))
#             return img
#         except Exception as e:
#             logging.info(f"Error reading image: {e}")
#             print("Error reading image:", e)
#             return None
        

# def extract_id_card(img):
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     blur = cv2.GaussianBlur(gray_img, (5,5),0)

#     thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     contours, _ = cv2.findContours(thresh ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     largest_contour = None
#     largest_area = 0

#     if not largest_contour.any():
#         return None
    
#     x, y, w, h = cv2.boundingRect(largest_contour)

#     logging.info(f"Contours are find at {(x, y, w, h)}")

#     current_wd = os.getcwd()
#     file_name = os.path.join(current_wd, intermediate_dir_path, conour_filename)
#     contour_id = img[y:y+h, x:x+w]
#     is_exists = file_exists(file_name)
#     if is_exists:
#         os.remove(file_name)

#     cv2.imwrite(file_name, contour_id)
#     return contour_id, file_name

# def save_image(image, filename, path="."):
#     full_path = os.path.join(path, filename)
#     is_exists = file_exists(full_path)
#     if is_exists:
#         os.remove(full_path)
    
#     cv2.imwrite(full_path, image)

#     logging.info(f"Image saved successfully: {full_path}")
#     return full_path


def read_image(image_path, is_uploaded=False):
    try:
        if is_uploaded:
            image_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_path)
        
        if img is None:
            raise Exception("Failed to read image.")
        
        return img
    except Exception as e:
        logging.error(f"Error reading image: {e}")
        print("Error reading image:", e)
        return None

def extract_id_card(img):
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise Exception("No contours found in the image.")

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        logging.info(f"Contours found at {(x, y, w, h)}")

        current_wd = os.getcwd()
        file_name = os.path.join(current_wd, intermediate_dir_path, contour_filename)
        contour_id = img[y:y+h, x:x+w]

        if file_exists(file_name):
            os.remove(file_name)

        cv2.imwrite(file_name, contour_id)
        return contour_id, file_name

    except Exception as e:
        logging.error(f"Error extracting ID card: {e}")
        print("Error extracting ID card:", e)
        return None, None

def save_image(image, filename, path="."):
    try:
        full_path = os.path.join(path, filename)
        if file_exists(full_path):
            os.remove(full_path)

        cv2.imwrite(full_path, image)
        logging.info(f"Image saved successfully: {full_path}")
        return full_path
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        print("Error saving image:", e)
        return None
