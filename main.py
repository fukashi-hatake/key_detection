from ultralytics import YOLO 
import torch 
import streamlit as st 
from PIL import Image 

model_name = "best.pt"

def main(): 
    mode = st.sidebar.selectbox("Model Type", ("Basic_Remote", "Basic_Remote_Tesla", "Tesla", "Key_Tesla")) 
    uploaded_file = st.file_uploader("Choose image: ")  

    if uploaded_file: 
        
        if mode == "Basic_Remote_Tesla": 
            model = torch.hub.load("ultralytics/yolov5", "custom", path=f"models/basic_remote_tesla.pt") 
        if mode == "Basic_Remote": 
            model = torch.hub.load("ultralytics/yolov5", "custom", path=f"models/basic_remote.pt") 
        if mode == "Tesla": 
            model = torch.hub.load("ultralytics/yolov5", "custom", path=f"models/tesla.pt") 
        if mode == "Key_Tesla": 
            model = torch.hub.load("ultralytics/yolov5", "custom", path=f"models/key_tesla.pt") 


        model = model.eval() 

        input_image = Image.open(uploaded_file)   

        st.write(input_image)

        result = model(input_image)

        result.show() 

        st.write(result.pandas().xyxy[0])


if __name__ == '__main__':
    main() 
