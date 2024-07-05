import streamlit as st
import cv2
import numpy as np

def add_gaussian_noise(image, mean, var):
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def apply_focus_blur(image, blur_strength, focus_area):
    center = (focus_area, focus_area)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, center, 2, (255), -1)
    mask_inv = cv2.bitwise_not(mask)
    blurred = cv2.GaussianBlur(image, (21, 21), blur_strength)
    focal_point = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
    combined = cv2.add(focal_point, background)
    return combined

def adjust_white_balance(image, temperature):
    adjustment = cv2.transform(image, np.array([[1, 0, 0], [0, 1-temperature, 0], [0, 0, 1+temperature]]))
    return adjustment

# Create a Streamlit app
st.title('Adding Noise In Synthetic Images Application')

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, 'Original Image')

    # Use session state to manage images
    if 'noisy_image' not in st.session_state:
        st.session_state.noisy_image = None
    if 'focused_image' not in st.session_state:
        st.session_state.focused_image = None

    # Gaussian Noise
    mean = st.slider('Gaussian Noise Mean', 0, 255, 50)
    var = st.slider('Gaussian Noise Variance', 0, 255, 25)
    if st.button('Apply Gaussian Noise'):
        st.session_state.noisy_image = add_gaussian_noise(image, mean, var)
        st.image([image, st.session_state.noisy_image], caption=['Original Image', 'With Gaussian Noise'])
        st.download_button("Download Noisy Image", data=cv2.imencode('.jpg', st.session_state.noisy_image)[1].tobytes(),
                           file_name='noisy_image.jpg', mime='image/jpg')

    # Focus Blur
    if st.session_state.noisy_image is not None:
        blur_strength = st.slider('Focus Blur Strength', 1, 100, 20)
        focus_area = st.slider('Focus Area', 1, image.shape[0]//2, image.shape[0]//4)
        if st.button('Apply Focus Blur'):
            st.session_state.focused_image = apply_focus_blur(st.session_state.noisy_image, blur_strength, focus_area)
            st.image([st.session_state.noisy_image, st.session_state.focused_image], caption=['Before Focus Blur', 'After Focus Blur'])
            st.download_button("Download Focused Image", data=cv2.imencode('.jpg', st.session_state.focused_image)[1].tobytes(),
                               file_name='focused_image.jpg', mime='image/jpg')

    # White Balance
    if st.session_state.focused_image is not None:
        temperature = st.slider('White Balance Temperature', -100.0, 100.0, 0.0)
        if st.button('Apply White Balance'):
            wb_adjusted = adjust_white_balance(st.session_state.focused_image, temperature)
            st.image([st.session_state.focused_image, wb_adjusted], caption=['Before White Balance', 'After White Balance'])
            st.download_button("Download White-Balanced Image", data=cv2.imencode('.jpg', wb_adjusted)[1].tobytes(),
                               file_name='white_balanced_image.jpg', mime='image/jpg')
