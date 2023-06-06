import streamlit as st
import time

def main():
    st.title("Sneaker Authenticator")
    st.write("Upload an image of your sneakers to determine authenticity.")
    
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        progress_bar = st.progress(0)

        for i in range(10):
            time.sleep(0.25)  # Simulate a delay
            progress_bar.progress((i + 1) * 10)
            
        # Display the uploaded image
         # Create a container to center align the image
        container = st.container()
        
        # Add CSS to center align the image
        container.markdown(
            """
            <style>
            .center-image {
                display: flex;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Display the uploaded image with a smaller size, centered
        container.image(uploaded_image, caption="Uploaded Image", width=300, 
                        clamp=False)
        
        # Call the prediction function to determine the brand and authenticity
        # brand, authenticity = predict_sneaker(uploaded_image)
        
        brand = "Nike"
        authenticity = "Authentic"

        # Display the brand and authenticity
        st.write(f"Brand: {brand} (Confidence Score: {85.74}%)")
        st.write(f"Authenticity: {authenticity} (Confidence Score: {96.96}%)")

if __name__ == "__main__":
    main()
