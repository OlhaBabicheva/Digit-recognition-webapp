from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torchvision
from model import Net

st.set_page_config(page_title = "Digit Recognition")

st.write('# MNIST Digit Recognition 🖋')
 
Network = Net(1, 10)
Network.load_state_dict(torch.load('model.pt'))
Network.eval()

sm = torch.nn.Softmax()
 
st.write('### Draw a digit between 0 and 9 in the black box below')
# Parameters for canvas in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)
 
realtime_update = st.sidebar.checkbox("Update in realtime", True)
 
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    update_streamlit=realtime_update,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    # Numpy array (4-channel RGBA 100,100,4)
    input_numpy_array = np.array(canvas_result.image_data)
    # RGBA PIL image
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image.save('user_input.png')
    # Convert to grayscale
    input_image_gs = input_image.convert('L')
    input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200,200)
    # Temporary image for opencv to read it
    input_image_gs.save('temp_for_cv2.jpg')
    image = cv2.imread('temp_for_cv2.jpg', 0)
    # Bounding box
    height, width = image.shape
    x,y,w,h = cv2.boundingRect(image)
    # Create new blank image and shift ROI to new coordinates
    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
    width, height = mask.shape
    x = width//2 - ROI.shape[0]//2
    y = height//2 - ROI.shape[1]//2
    mask[y:y+h, x:x+w] = ROI
    output_image = Image.fromarray(mask) # mask has values in [0-255] as expected
    compressed_output_image = output_image.resize((22,22), Image.NEAREST)
 
    convert_tensor = torchvision.transforms.ToTensor()
    tensor_image = convert_tensor(compressed_output_image)
    tensor_image = tensor_image/255.
    # Padding
    tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
    img = Image.fromarray(tensor_image.detach().cpu().numpy().reshape(28,28), mode='L')
    img.save("processed_tensor.png", "PNG")
    # So we use matplotlib to save it instead
    plt.imsave('processed_tensor.png',tensor_image.detach().cpu().numpy().reshape(28,28), cmap='gray')
 
     
    device='cpu'
    with torch.no_grad():
        output0 = Network(torch.unsqueeze(tensor_image, dim=0).to(device=device))
        _, output = torch.max(output0[0], 0)
        probabilities, output1 = torch.topk(output0[0], 3)
        probabilities = sm(probabilities)

    st.write('### Prediction') 
    st.write(f'### {output}')
    st.write('## Breakdown of the prediction process:')
 
    st.write('### Image as a grayscale Numpy array')
    st.write(input_image_gs_np)
 
    st.write('### Processed image')
    st.image('processed_tensor.png')
 
 
 
    st.write('### Prediction') 
    st.write(f'{output.item()}')
    st.write('### Certainty')
    st.write(f'{round(probabilities[0].item(), 2)*100}%')
    st.write('### Top 3 candidates')
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.bar([0, 1, 2], probabilities, color='#FF4D4B')
    plt.xticks([0, 1, 2], [num.item() for num in output1])
    st.pyplot(fig)
    st.write('### Certainties')    
    st.write(f'''{round(probabilities[0].item(), 2)*100}%  
             {round(probabilities[1].item(), 2)*100}%  
             {round(probabilities[2].item(), 2)*100}%''')