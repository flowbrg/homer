import streamlit as st
from PIL import Image


############################## Page ##############################


st.markdown("""
  <br>
  <br>
  
  
  """,unsafe_allow_html=True)
# Image on the left, text on the right
col1, col2 = st.columns([1, 2])  # [1,2] = Image takes 1/3, text 2/3

with col1:
  with Image.open("static/homerlogo-nobg.png") as logo:
    st.image(logo, width=150)

with col2:
  st.markdown("""
  ### Welcome to HOMER  
  <br>
  <div style="display: flex; align-items: center; gap: 15px;">
    <span>Start by uploading your documents</span>
    <a href="./index" target="_self">
      <button style="padding:0.3em 0.8em; font-size:16px; background-color:#512967; color:white; border:none; border-radius:5px; cursor:pointer;">
        HERE
      </button>
    </a>
  </div>
  
  <br>
  Then, you can either ask a simple question or generate a full structured report based on the content of your files.
  """, unsafe_allow_html=True)


############################## Footer ##############################


st.markdown("""
<style>
.footer {
position: fixed;
bottom: 0;
left: 0;
width: 100%;
text-align: center;
font-size: 12px;
color: black;
background-color: #f5f5f5;
padding: 10px 0;
border-top: 1px solid #ddd;
z-index: 100;
}
</style>

<div class="footer">
Designed by Florent Bergé & Mathieu de la Barre (IMT Atlantique) for SCK CEN
</div>
""", unsafe_allow_html=True)