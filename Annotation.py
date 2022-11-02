import numpy as np
import cv2
import streamlit as st
from PIL import Image


st.title("Blur Filters")

image=st.sidebar.file_uploader(
    "Select Image",
    type=("png","jpg","jpeg")
)