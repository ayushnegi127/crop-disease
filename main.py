import streamlit as st
import tensorflow as tf
import numpy as np
import random


st.set_page_config(page_title="Plant Disease Detection", page_icon=":leaves:")


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Function to extract disease name
def extract_disease_name(full_class_name):
    if '___' in full_class_name:
        return full_class_name.split('___')[1]
    else:
        return full_class_name  # For classes like Soybean___healthy


# List of random quick facts
quick_facts = [
    "Did you know? Fungal diseases cause around 85% of plant diseases globally!",
    "Interesting fact: Proper crop rotation can reduce the risk of soil-borne diseases.",
    "Quick tip: Always water plants at their base to prevent leaf diseases.",
    "Fun fact: Some plants produce natural antifungal chemicals to protect themselves from disease.",
    "Did you know? Many plant diseases are spread by insects like aphids and whiteflies.",
    "Remember: Healthy soil promotes healthy plants, reducing susceptibility to diseases."
]

# Disease descriptions and solutions
disease_info = {
    'Apple_scab': {
        'description': "Apple scab is a fungal disease that causes dark, sunken lesions on the fruit, leaves, and stems of apple trees.",
        'solution': "Prune infected branches and apply fungicides. Ensure good air circulation around trees."
    },
    'Black_rot': {
        'description': "Black rot is a fungal disease that causes dark, sunken lesions on apple fruit and leaves.",
        'solution': "Remove and destroy infected fruit and leaves. Apply fungicides as a preventive measure."
    },
    'Cedar_apple_rust': {
        'description': "Cedar apple rust is a fungal disease that causes orange, rust-colored spots on apple leaves and fruit.",
        'solution': "Remove cedar trees nearby, and apply fungicides during the growing season."
    },
    'healthy': {
        'description': "The plant appears healthy and free from visible disease symptoms.",
        'solution': "No action required."
    },
    'Powdery_mildew': {
        'description': "Powdery mildew is a fungal disease that causes a white powdery coating on the leaves and stems of plants.",
        'solution': "Increase air circulation and apply fungicides. Avoid overhead watering."
    },
    'Cercospora_leaf_spot': {
        'description': "Cercospora leaf spot is a fungal disease that causes dark, circular spots on leaves.",
        'solution': "Remove and destroy infected leaves and apply fungicides."
    },
    'Common_rust_': {
        'description': "Common rust is a fungal disease that causes reddish-brown pustules on maize leaves.",
        'solution': "Use resistant varieties and apply fungicides if necessary."
    },
    'Northern_Leaf_Blight': {
        'description': "Northern leaf blight is a fungal disease that causes long, grayish-green lesions on maize leaves.",
        'solution': "Remove infected leaves and apply fungicides."
    },
    'Grape_Black_rot': {
        'description': "Black rot is a fungal disease that causes dark, sunken lesions on grape fruit and leaves.",
        'solution': "Prune and destroy infected plant parts and apply fungicides."
    },
    'Esca_(Black_Measles)': {
        'description': "Esca is a fungal disease that causes a variety of symptoms including leaf spots and decay in grapevines.",
        'solution': "Prune affected areas and apply fungicides. Ensure proper vineyard management."
    },
    'Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': "Leaf blight is a disease that causes large, irregularly shaped spots on plant leaves.",
        'solution': "Remove and destroy infected leaves. Apply appropriate fungicides."
    },
    'Haunglongbing': {
        'description': "Huanglongbing (HLB) is a bacterial disease that causes yellowing and deformity of citrus fruit and leaves.",
        'solution': "Remove infected trees and apply insecticides to control the vector."
    },
    'Bacterial_spot': {
        'description': "Bacterial spot causes dark, water-soaked lesions on fruit and leaves of pepper and tomato plants.",
        'solution': "Remove and destroy infected plant parts and apply copper-based bactericides."
    },
    'Early_blight': {
        'description': "Early blight is a fungal disease that causes dark, concentric rings on tomato and potato leaves.",
        'solution': "Rotate crops and apply fungicides as a preventive measure."
    },
    'Late_blight': {
        'description': "Late blight is a fungal disease that causes a white, downy mildew on the undersides of leaves, leading to rapid plant decay.",
        'solution': "Remove and destroy infected plants and apply fungicides."
    },
    'Leaf_Mold': {
        'description': "Leaf mold is a fungal disease that causes a grayish mold on the upper surfaces of tomato leaves.",
        'solution': "Improve air circulation and apply fungicides."
    },
    'Septoria_leaf_spot': {
        'description': "Septoria leaf spot is a fungal disease that causes small, round spots with dark edges on tomato leaves.",
        'solution': "Remove and destroy infected leaves and apply fungicides."
    },
    'Spider_mites_Two-spotted_spider_mite': {
        'description': "Spider mites are tiny pests that cause stippling and webbing on plant leaves, leading to a decrease in plant health.",
        'solution': "Use miticides or insecticidal soaps and ensure proper irrigation."
    },
    'Target_Spot': {
        'description': "Target spot is a fungal disease that causes concentric rings on tomato leaves, giving a target-like appearance.",
        'solution': "Apply fungicides and improve crop rotation practices."
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'description': "Tomato yellow leaf curl virus causes yellowing and curling of tomato leaves, affecting plant growth.",
        'solution': "Use resistant tomato varieties and control whiteflies, the primary vector."
    },
    'Tomato_mosaic_virus': {
        'description': "Tomato mosaic virus causes mottling and distortion of tomato leaves and fruits.",
        'solution': "Use virus-free seeds and control aphids."
    },
    'Leaf_scorch': {
        'description': "Leaf scorch is a condition that causes the edges of leaves to turn brown and dry out, often due to environmental stress like drought or excessive heat.",
        'solution': "Ensure consistent watering and provide shade during extreme heat. Improve soil moisture and avoid over-fertilizing."
    }
    
}

# Sidebar
st.sidebar.title("Dashboard 🌿")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Machine Learning for Detection and Prediction of Crop Disease and Pests")
    image_path = "images/disease.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Crop Disease Recognition System! 🌿🔍
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    image_path = "images/wall3.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

elif app_mode == "Disease Recognition":
    col1, col2 = st.columns([1, 5])  # Adjust the width ratio as needed
    
    with col1:
        image_path = "images/logo.png"
        st.image(image_path, width=100)  # Minimized image size
    
    with col2:
        st.header("Disease Recognition")
    
    st.write("Please upload an image of a plant, leaf, or crop to get predictions. **Ensure the image is of a plant or related crop. Human or irrelevant images may result in inaccurate predictions.**")
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, width=400, use_column_width=True)
        
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

            
            predicted_disease = extract_disease_name(class_name[result_index])
            st.success(f"Model is Predicting: {predicted_disease}")

            disease_info_dict = disease_info.get(predicted_disease, {"description": "No description available.", "solution": "No solution available."})
            st.write(f"**Description:** {disease_info_dict['description']}")
            st.write(f"**Solution:** {disease_info_dict['solution']}")
            
            # Show a random quick fact
            random_fact = random.choice(quick_facts)
            st.info(f"🌿 Quick Fact: {random_fact}")
    else:
        st.warning("Please upload an image first.")
        st.button("Predict", disabled=True)