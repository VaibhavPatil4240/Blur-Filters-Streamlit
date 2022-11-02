import numpy as np
import cv2
import streamlit as st
from PIL import Image





@st.cache
def averageFilter(img,kernel=(1,1)):
        return cv2.blur(np.copy(img),kernel)
@st.cache
def boxFilter(img,kernel=(1,1)):
        return cv2.boxFilter(src=img,ksize=kernel,ddepth=-1,normalize=False)
@st.cache
def guassianFilter(img,kernel=(1,1)):
        return cv2.GaussianBlur(img,kernel,0)
@st.cache
def medianFilter(img,kernel=1):
        return cv2.medianBlur(img,kernel,0)


st.title("Blur Filters")

image=st.sidebar.file_uploader(
    "Select Image",
    type=("png","jpg","jpeg")
)
filteredImage=None
numpy_image=None

if not image==None:
    numpy_image=np.asarray(Image.open(image))

imageCanvas=st.empty()



filter=st.radio(
    "Select Filter",
    ("Original","Average Filter","Box Filter","Guassian Filter","Median Filter"),
    horizontal=True,
    key="filterval"
)

slider=st.slider(
    label="Kernel",
    max_value=100,
    step=2,
    min_value=1
)
kernel=(slider,slider)
if (filter=="Average Filter") and not numpy_image is None:
    filteredImage=averageFilter(numpy_image,kernel)
elif filter=="Original":
    filteredImage=image
elif filter=="Box Filter" and not numpy_image is None:
    filteredImage=boxFilter(numpy_image,kernel)
elif filter=="Guassian Filter" and not numpy_image is None:
    filteredImage=guassianFilter(numpy_image,kernel)
elif filter=="Median Filter" and not numpy_image is None:
    filteredImage=medianFilter(numpy_image,slider)

if(not filteredImage is None):
    imageCanvas.image(filteredImage,caption=filter,width=450)
