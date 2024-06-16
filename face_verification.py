import face_recognition
from deepface import DeepFace
import numpy as np
import cv2
import os
import logging
from utils import file_exists, read_yaml

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="s")

config_path = "config.yaml"
config = read_yaml(config_path)

artifacts = config['artifacts']
cascade_path = artifacts['HAARCASCADE_PATH']
output_path = artifacts['INTERMEDIATE_DIR']

# def detect_and_extract_face(img):
#     gray_img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     face_cacade = cv2.CascadeClassifier(cascade_path)

#     faces = face_cacade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

#     max_area = 0
#     largest_face = None
#     for (x, y, w, h) in faces:
#         area = w*h
#         if area>max_area:
#             max_area = area
#             largest_face = (x, y, w, h)

#     if largest_face is not None:
#         (x, y, w, h) = largest_face

#         new_w = int(w * 1.50)
#         new_h = int(h * 1.50)

#         new_x = max(0, x - int((new_w - w)/2))
#         new_y = max(0, y - int((new_h - h)/2))

#         extracted_face = img[new_y:new_y+new_h, new_x:new_x+new_w]

#         current_wd  = os.getcwd()
#         filename = os.path.join(current_wd, output_path, "extended_face.jpg")

#         if os.path.exists(filename):
#             os.remove(filename)

#         cv2.imwrite(filename, extracted_face)
#         print(f"Extracted face saved at: {filename}")

#     else:
#         return None

def detect_and_extract_face(img):
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            raise Exception("Failed to load Haar cascade XML file from path: " + cascade_path)
        
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            logging.info("No face detected in the image.")
            return None

        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
        extracted_face = img[y:y+h, x:x+w]

        filename = os.path.join(output_path, "extracted_face.jpg")
        if os.path.exists(filename):
            os.remove(filename)
        
        cv2.imwrite(filename, extracted_face)
        logging.info(f"Extracted face saved at: {filename}")
        return filename
    
    except Exception as e:
        logging.error(f"Error detecting and extracting face: {e}")
        print(f"Error detecting and extracting face: {e}")
        return None
    
def face_recognition_face_comparision(image1_path="data\\02_intermediate_data\\extracted_face.jpg", image2_path="data\\02_intermediate_data\\face_image.jpg"):
    img1_exists = file_exists(image1_path)
    img2_exists = file_exists(image2_path)

    if img1_exists and img2_exists:
        print("Checking the path for the images provided")
        return False
    
    image1 = face_recognition.load_image_file(image1_path)
    image2 = face_recognition.load_image_file(image2_path)

    if image1 is not None and image2 is not None:
        face_encodings1 = face_recognition.face_encodings(image1)
        face_encodings2 = face_recognition.face_encodings(image2)

    else:
        print("Image is not loaded properly")
        return False
    
    if len(face_encodings1)==0 or len(face_encodings2)==0:
        print("No face detected in one or both images.")
        return False
    
    else:
        matches = face_recognition.compare_faces(np.array(face_encodings1), np.array(face_encodings2))

    if matches[0]:
        print("Faces are verified")
        return True
    else:
        return False
    

def deepface_face_comparison(image1_path="data\\02_intermediate_data\\extracted_face.jpg", image2_path = "data\\02_intermediate_data\\face_image.jpg"):
    img1_exists = file_exists(image1_path)
    img2_exists = file_exists(image2_path)

    if not(img1_exists or img2_exists):
        print("Check the path for the images provided")
        return False
    
    verification = DeepFace.verify(img1_path=image1_path, img2_path=image2_path)

    if len(verification) > 0 and verification['verified']:
        print("faces are verified")
        return True
    else:
        return False
    
def face_comparison(image1_path, image2_path, model_name = 'deepface'):
    is_verified = False
    if model_name == 'deepface':
        is_verified = deepface_face_comparison(image1_path, image2_path)
    elif model_name == 'facerecognition':
        is_verified = face_recognition_face_comparision(image1_path, image2_path)
    else:
        print("Mention proper model name for face recognition")

    return is_verified

def get_face_embeddings(image_path):
    img_exists = file_exists(image_path)

    if not (img_exists):
        print("Check the path for the images provided")
        return None
    
    embedding_objs = DeepFace.represent(img_path = image_path, model_name= "Facenet")
    embedding = embedding_objs[0]["embedding"]

    if len(embedding) > 0:
        return embedding

    return None

if __name__ == "__main__":
        id_card = ""
        extracted_face_path= detect_and_extract_face(image_path=id_card)
        


