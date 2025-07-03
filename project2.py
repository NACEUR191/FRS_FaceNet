import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from keras_facenet import FaceNet  

# Initialize FaceNet embedder
embedder = FaceNet()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("🔍 Face Recognition System with FaceNet")

# Session states
if 'faces' not in st.session_state:
    st.session_state.faces = []
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []
if 'model' not in st.session_state:
    st.session_state.model = None

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Step 1: Upload image and detect faces
st.header("Step 1: Upload Image and Detect Faces")
img_file = st.file_uploader("Upload an image (jpg/png)", type=['jpg', 'png'])
label = st.text_input("Label (person's name)")

if img_file and label:
    img = Image.open(img_file)
    img_np = np.array(img.convert('RGB'))
    faces = detect_faces(img_np)
    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            face_img = img_np[y:y+h, x:x+w]
            st.image(face_img, caption=f"Face for label: {label}", use_column_width=True)
            st.session_state.faces.append(face_img)
            st.session_state.labels.append(label)

# Step 2: Extract embeddings using FaceNet
st.header("Step 2: Extract Features with FaceNet")
if st.button("Extract Features") and st.session_state.faces:
    for face in st.session_state.faces[len(st.session_state.embeddings):]:  # Extract only new faces
        # FaceNet expects RGB images resized to 160x160
        face_resized = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face_resized])[0]
        st.session_state.embeddings.append(embedding)
    st.success("Features extracted for all faces!")

# Step 3: Train SVM classifier
st.header("Step 3: Train Recognition Model")
if st.button("Train Model") and st.session_state.embeddings:
    X = np.array(st.session_state.embeddings)
    y_labels = st.session_state.labels
    if len(X) != len(y_labels):
        st.error(f"Mismatch: {len(X)} embeddings vs {len(y_labels)} labels.")
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_labels)
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X, y)
        st.session_state.model = clf
        st.session_state.le = le
        st.success("Model trained successfully!")

        # Save embeddings and model to files
        np.savez_compressed('embeddings_labels.npz', embeddings=X, labels=y_labels)
        joblib.dump(clf, 'svm_face_recognition_model.joblib')
        joblib.dump(le, 'label_encoder.joblib')

        st.download_button("Download Embeddings & Labels (.npz)", data=open('embeddings_labels.npz', 'rb').read(), file_name='embeddings_labels.npz')
        st.download_button("Download SVM Model (.joblib)", data=open('svm_face_recognition_model.joblib', 'rb').read(), file_name='svm_face_recognition_model.joblib')
        st.download_button("Download Label Encoder (.joblib)", data=open('label_encoder.joblib', 'rb').read(), file_name='label_encoder.joblib')

# Step 4: Recognition on new image
st.header("Recognition Test")
test_img = st.file_uploader("Upload a test image", type=['jpg', 'png'], key='test_img')
if test_img and st.session_state.model:
    test_img_pil = Image.open(test_img)
    test_np = np.array(test_img_pil.convert('RGB'))
    faces = detect_faces(test_np)
    if len(faces) == 0:
        st.warning("No face detected in test image.")
    else:
        for (x, y, w, h) in faces:
            face_crop = test_np[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (160, 160))
            embedding = embedder.embeddings([face_resized])[0]
            pred = st.session_state.model.predict([embedding])[0]
            label = st.session_state.le.inverse_transform([pred])[0]
            st.image(face_crop, caption=f"Predicted: {label}", use_column_width=True)
