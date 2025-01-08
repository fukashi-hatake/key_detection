from ultralytics import YOLO 
import torch 
import streamlit as st 
from PIL import Image 

model_name = "best.pt"

def main(): 
    mode = st.sidebar.selectbox("Model Type", ("Key Detection", "Vin/Odometer")) 
    uploaded_file = st.file_uploader("Choose image: ")  

    if uploaded_file: 
        
        if mode == "Vin/Odometer": 
            model = torch.hub.load("ultralytics/yolov5", "custom", path=f"models/{model_name}") 
        else: 
            model = torch.hub.load("ultralytics/yolov5", "custom", path=f"models/basic_remote.pt") 

        model = model.eval() 

        input_image = Image.open(uploaded_file)   

        st.write(input_image)

        result = model(input_image)

        result.show() 

        st.write(result.pandas().xyxy[0])


if __name__ == '__main__':
    main() 
