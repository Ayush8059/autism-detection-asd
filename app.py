import streamlit as st
from PIL import Image
import io
import sys
import os

# Add the current directory to the path to import helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from helper import create_activity_database, ActivityRecommender, load_prediction_model, predict_from_image
except ImportError as e:
    st.error(f"Failed to import helper functions: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Autism Support System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Caching ---
@st.cache_resource
def initialize_system():
    """Load and cache the model and recommender to prevent reloading on each interaction."""
    model = load_prediction_model()
    activity_db = create_activity_database()
    recommender = ActivityRecommender(activity_db)
    return model, recommender

# --- Main Application ---
def main():
    st.title("🧩 Autism Support System")

    # --- Sidebar for Status and Information ---
    with st.sidebar:
        st.header("System Status")
        model, recommender = initialize_system()
        
        if model is None:
            st.error("❌ Model loading failed.")
            st.error("The prediction model could not be loaded. Please ensure the 'autism_detection_vgg16_finetuned.h5' file is available.")
            st.info("For demonstration purposes, the app will continue with a mock model.")
            # Create a mock model flag for demonstration
            st.session_state.mock_model = True
        else:
            st.success("✅ Model loaded successfully.")
            st.session_state.mock_model = False

        st.divider()
        st.info("""
        **About this tool:**
        - Upload a child's facial image for analysis
        - Get activity recommendations based on needs
        - Focuses on various developmental areas
        """)
        
        st.divider()
        st.warning(
            "**Disclaimer:** This is an informational tool and not a substitute for professional medical advice. Always consult a qualified health provider."
        )

    # --- Initialize Session State ---
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
        st.session_state.prediction_score = None

    # --- Step 1: Image Upload ---
    st.header("1. Upload an Image")
    st.info("Please upload a clear, front-facing image of a child's face.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        
        # Display the uploaded image
        try:
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
            st.stop()

        if st.button("Analyze Image", use_container_width=True, type="primary"):
            with st.spinner('Analyzing the image...'):
                if st.session_state.mock_model:
                    # Mock prediction for demonstration
                    st.session_state.prediction_result = "Autistic"
                    st.session_state.prediction_score = 0.3
                    st.success("Analysis complete (using mock data)")
                else:
                    image_bytes = io.BytesIO(image_data)
                    prediction, score = predict_from_image(image_bytes, model)
                    # Store results in session state
                    st.session_state.prediction_result = prediction
                    st.session_state.prediction_score = score

    # --- Step 2: Display Prediction and Gather Input ---
    if st.session_state.prediction_result:
        prediction = st.session_state.prediction_result
        score = st.session_state.prediction_score

        st.header("2. Prediction Result")
        
        if "Error" in prediction:
            st.error(prediction)
            st.info("Please try again with a different image.")
        elif prediction == "Autistic":
            confidence = (1 - score) * 100
            st.success(f"**Prediction: {prediction}** (Confidence: {confidence:.2f}%)")

            # --- Human Input for Recommendation ---
            st.header("3. Select Areas for Support")
            st.info("Based on the prediction, please select the areas where you'd like activity recommendations.")
            
            # Use the same categories as in the helper file
            available_needs = [
                "sensory_integration", "fine_motor", "calming", "creative",
                "gross_motor", "social_skills", "communication",
                "emotional_regulation", "concentration", "problem_solving",
                "vestibular", "proprioceptive", "energy_release", "hand_eye_coordination",
                "turn_taking", "rules_following", "comprehension", "oral_motor",
                "joint_attention", "non_verbal", "requesting", "choice_making",
                "auditory"
            ]
            
            user_choices = st.multiselect(
                "Select one or more areas:",
                options=sorted(available_needs),
                help="Choose the developmental areas you want to focus on"
            )

            if st.button("Get Activity Recommendations", use_container_width=True, type="primary"):
                if not user_choices:
                    st.warning("Please select at least one area to get recommendations.")
                else:
                    needs_string = " ".join(user_choices)
                    with st.spinner("Generating personalized recommendations..."):
                        recommendations = recommender.recommend(needs_string)
                        
                        st.header("4. Recommended Activities")
                        if not recommendations.empty:
                            for index, row in recommendations.iterrows():
                                with st.expander(f"{row['name']} (Ages {row['age_range']})"):
                                    st.markdown(f"**Focuses on:**")
                                    skills = row['skills_targeted'].split()
                                    for skill in skills:
                                        st.markdown(f"- {skill.replace('_', ' ').title()}")
                        else:
                            st.info("No specific activities found for the selected combination. Try selecting fewer or different options.")

        else:  # Non_Autistic
            confidence = score * 100
            st.info(f"**Prediction: {prediction}** (Confidence: {confidence:.2f}%)")
            st.markdown("Based on this prediction, no specific support activities are recommended at this time.")
            st.markdown("If you have concerns about a child's development, please consult with a healthcare professional.")

    # --- Add informational section at the bottom ---
    st.divider()
    with st.expander("ℹ️ About Autism Spectrum Disorder"):
        st.markdown("""
        Autism Spectrum Disorder (ASD) is a developmental disability that can cause significant social, 
        communication and behavioral challenges. People with ASD may communicate, interact, behave, 
        and learn in ways that are different from most other people.
        
        **Early signs of ASD may include:**
        - Not responding to their name by 12 months
        - Not pointing at objects to show interest by 14 months
        - Avoiding eye contact
        - Having trouble understanding other people's feelings
        - Delayed speech and language skills
        - Repetitive behaviors
        - Unusual reactions to sensory input
        """)

if __name__ == "__main__":
    main()