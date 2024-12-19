import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the enhanced model with caching to prevent repeated loading attempts
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(r'C:\Users\PRATYUSH\Desktop\Digit_recognizer\mnist_model_enhanced.h5')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Failed to load model. Please check the model file and try again. Error: {str(e)}")
        return None

# Load the model
model = load_model()

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        padding-top: 30px;
        text-shadow: 2px 2px 4px #000000;
    }
    .subtitle {
        font-size: 22px;
        color: #000000;
        text-align: center;
        margin-bottom: 40px;
        font-style: italic;
    }
    .st-canvas {
        border: 3px solid #FF6347;
        border-radius: 15px;
        margin: 0 auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .prediction-box {
        font-size: 40px;
        color: #32CD32;
        text-align: center;
        font-weight: bold;
        padding: 25px;
        background-color: #FAFAD2;
        border-radius: 15px;
        margin-top: 30px;
    }
    .confidence-text {
        font-size: 20px;
        color: #20B2AA;
        text-align: center;
        font-style: italic;
    }
    .footer {
        font-size: 16px;
        color: #FFFFFF;
        text-align: center;
        margin-top: 60px;
        font-weight: bold;
    }
    .prediction-probabilities {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and instructions with emojis
st.markdown('<div class="title"> Handwritten Digit Recognition </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a digit below, and our AI will guess it with confidence! </div>', unsafe_allow_html=True)

# Create a centered layout for the canvas and predictions
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Create a canvas for user input
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# Proceed with predictions only if the model is loaded successfully
if model:
    # Check for image data on the canvas before proceeding
    if canvas_result.image_data is not None:
        # Check if the canvas is not completely blank
        if np.sum(canvas_result.image_data) > 0:
            if st.button("Predict Digit"):
                try:
                    # Preprocess the image from the canvas
                    img = canvas_result.image_data
                    img = cv2.resize(img, (28, 28))  # Resize to 28x28
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    img = img / 255.0  # Normalize
                    img = np.expand_dims(img, axis=(0, -1))  # Reshape to (1, 28, 28, 1)

                    st.markdown('<div class="prediction-probabilities"><h3>🖼️ Processed Input Image:</h3></div>', unsafe_allow_html=True)
                    st.image(img.squeeze(), width=150)

                    # Predict the digit
                    prediction = model.predict(img)

                    # Check if prediction is valid
                    if prediction.size > 0:
                        predicted_class = np.argmax(prediction, axis=1)[0]
                        confidence = np.max(prediction) * 100

                        # Display the prediction with a custom style
                        st.markdown(f'<div class="prediction-box">Predicted Digit: {predicted_class}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="confidence-text">Confidence Level: {confidence:.2f}%</div>', unsafe_allow_html=True)

                        # Display the prediction probabilities bar chart
                        st.markdown("### 🔢 Prediction Probabilities")
                        fig, ax = plt.subplots()
                        bars = ax.bar(range(10), prediction[0], color="#4682B4", edgecolor="#4682B4")
                        bars[predicted_class].set_color("#32CD32")  # Highlight the predicted digit
                        ax.set_xticks(range(10))
                        ax.set_xlabel("Digit", fontsize=12, fontweight='bold')
                        ax.set_ylabel("Probability", fontsize=12, fontweight='bold')
                        ax.set_title("Model Confidence per Digit", fontsize=16, fontweight='bold', color="#4B0082")
                        st.pyplot(fig)
                    else:
                        st.error("Prediction returned empty. Please check the model and input data.")

                except IndexError as e:
                    st.error("An error occurred: Tried to access an index that doesn't exist.")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.warning("Please draw a digit on the canvas to get a prediction.")
else:
    st.warning("The model could not be loaded, so predictions are unavailable at this time.")

# Footer with developer name
st.markdown(
    """
    <style>
    .footer {
        font-size: 15px;
        color: #000000;  /* Black color */
        text-align: center;
        margin-top: 50px;
    }
    .footer-line {
        font-size: 30px;
        font-weight: bold;
    }
    .footer-name {
        font-size: 50px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Footer content with two lines
st.markdown('<div class="footer"><div class="footer-line">Brought to you by</div><div class="footer-name">Pratyush Prayash Jena</div></div>', unsafe_allow_html=True)


