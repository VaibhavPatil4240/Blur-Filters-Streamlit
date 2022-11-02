import numpy as np
import cv2
import streamlit as st
from PIL import Image

@st.cache
def sharpen(img,kernel=1):
    kernel=np.array([[0, -1, -0], [-1, kernel, -1], [0, -1, 0]])
    return cv2.filter2D(img,-1,kernel)
@st.cache
def emboss(image):
    kernel = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
    return cv2.filter2D(image, -1, kernel)

@st.cache
def HDR(img,sigma_s,sigma_r):
    if len(img.shape) <3 :
        return img
    return cv2.detailEnhance(img,sigma_s,sigma_r)

@st.cache
def sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]])
    return cv2.filter2D(img, -1, kernel)

@st.cache
def brigthness(img,value):
    h,s,v=cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2HSV))
    v=cv2.add(v,value)
    return cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2RGB)
@st.cache
def averageFilter(img,kernel=(1,1)):
        return cv2.blur(np.copy(img),kernel)
@st.cache
def guassianFilter(img,kernel=(1,1)):
        return cv2.GaussianBlur(img,kernel,0)
@st.cache
def medianFilter(img,kernel=1):
        return cv2.medianBlur(img,kernel,0)
st.title("Instagram Filters")

@st.cache
def paint(img,sigma_s,sigma_r):
    return cv2.stylization(img, sigma_s, sigma_r)

image=st.file_uploader(
    "Select Image",
    type=("png","jpg","jpeg")
)

imageCanvas=st.empty()

blackNWhite=st.sidebar.checkbox("Black and White")

width= st.sidebar.slider(
    "Image size",
    min_value=100,
    max_value=1000,
    value=300
)
sharpenKernel=st.sidebar.slider(
    "Sharpen",
    min_value=0,
    max_value=10
)
brigthnessValue=st.sidebar.slider(
    "Brigthness",
    min_value=-255,
    max_value=255,
    value=0
)

smoothFilter=st.sidebar.selectbox(
    "Smoothness Filter",
    options=("Guassian Blur","Average Blur","Median Blur")
)
smothnessIntensity=st.sidebar.slider(
    label="Intensity",
    max_value=100,
    step=2,
    min_value=1
)

paintBox=st.sidebar.checkbox("Painting")
paintIntensity=st.sidebar.slider(
    label="Intensity",
    max_value=10,
    min_value=1,
    key="paintIntensity"
)
paintStrength=st.sidebar.slider(
    label="Strength",
    max_value=100,
    min_value=1,
    key="paintStrength"
)
hdrBox=st.sidebar.checkbox("HDR")
hdrIntensity=st.sidebar.slider(
    label="Intensity",
    max_value=10,
    min_value=1
)
hdrStrength=st.sidebar.slider(
    label="Strength",
    max_value=100,
    min_value=1
)

sepiaBox=st.sidebar.checkbox("Sepia")
embossBox=st.sidebar.checkbox("Emboss")

numpy_image=None
if(not image is None):
    numpy_image= np.asanyarray(Image.open(image),'uint8')
if(sharpenKernel>0 and not numpy_image is None):
    numpy_image=sharpen(numpy_image,sharpenKernel)

if(not numpy_image is None):
    numpy_image=brigthness(numpy_image,brigthnessValue)

if(blackNWhite and not numpy_image is None):
    numpy_image=cv2.cvtColor(numpy_image,cv2.COLOR_RGB2GRAY)

if(not numpy_image is None):
    kernel=(smothnessIntensity,smothnessIntensity)
    if(smoothFilter=="Guassian Blur"):
        numpy_image=guassianFilter(numpy_image,kernel)
    elif(smoothFilter=="Average Blur"):
        numpy_image=averageFilter(numpy_image,kernel)
    elif(smoothFilter=="Median Blur"):
        numpy_image=medianFilter(numpy_image,smothnessIntensity)

if(paintBox and not numpy_image is None):
    numpy_image=paint(numpy_image,paintStrength,paintIntensity/10)

if(hdrBox and not numpy_image is None):
    numpy_image=HDR(numpy_image,hdrStrength,hdrIntensity/10)

if(sepiaBox and not numpy_image is None):
    numpy_image=sepia(numpy_image)
if(embossBox and not numpy_image is None):
    numpy_image=emboss(numpy_image)

if(not numpy_image is None):
    imageCanvas.image(numpy_image,caption="Filtered Image",width=width)
