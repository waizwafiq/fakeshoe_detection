import streamlit as st
import time

import torch
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layer 1: 3 input channels, 16 output channels, 3x3 kernel size, stride of 1, padding of 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with 2x2 kernel and stride of 2

        # Convolutional layer 2: 16 input channels, 32 output channels, 3x3 kernel size, stride of 1, padding of 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with 2x2 kernel and stride of 2

        # Fully connected layer 1: Input size 32*56*56 (224/2^2), output size 64
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.relu3 = nn.ReLU()

        # Fully connected layer 2: Input size 64, output size 2
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)  # Apply convolutional layer 1
        x = self.relu1(x)  # Apply ReLU activation function
        x = self.pool1(x)  # Apply max pooling

        x = self.conv2(x)  # Apply convolutional layer 2
        x = self.relu2(x)  # Apply ReLU activation function
        x = self.pool2(x)  # Apply max pooling

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc1(x)  # Apply fully connected layer 1
        x = self.relu3(x)  # Apply ReLU activation function
        x = self.fc2(x)  # Apply fully connected layer 2

        return x

def predict_brand(image):
    # Load the saved model from file
    model = CNN()  # Instantiate your model
    model.load_state_dict(torch.load('./models/brand_detect_modelv1.pth'))

    model.eval()

    # Apply the transformation to the image
    image = transform(image).unsqueeze(0)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
    
    # Return the predicted class and probability score
    if predicted_class.item() == 0:
        shoe_brand = "Adidas"
    else:
        shoe_brand = "Nike"

    return shoe_brand, predicted_prob.item()

def predict_auth(image):
    # Load the saved model from file
    model = CNN()  # Instantiate your model
    model.load_state_dict(torch.load('./models/fake_detect_modelv1.pth'))

    model.eval()

    # Apply the transformation to the image
    image = transform(image).unsqueeze(0)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
    
    # Return the predicted class and probability score
    if predicted_class.item() == 0:
        authenticity = "Authentic"
    else:
        authenticity = "Counterfeit"

    return authenticity, predicted_prob.item()



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
        
        image = Image.open(uploaded_image)
        
        # Call the prediction function to determine the brand and authenticity
        # brand, authenticity = predict_sneaker(uploaded_image)
        
        brand = predict_brand(image)
        authenticity = predict_auth(image)

        # Display the brand and authenticity
        st.write(f"Brand: {brand[0]} (Confidence Score: {brand[1]*100:.3f}%)")
        st.write(f"Authenticity: {authenticity[0]} (Confidence Score: {authenticity[1]*100:.3f}%)")

if __name__ == "__main__":
    main()
