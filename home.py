import streamlit as st
from PIL import Image


from src.core.configuration import load_config

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    #if "backend" not in st.session_state:
    #    st.session_state.backend = Application(config=st.session_state.baseConfig)


def main():
    logo = Image.open("./static/homerlogo-nobg.png")  # Remplace par le bon chemin si besoin
    st.markdown("""
        <br>
        <br>
        
        
        """,unsafe_allow_html=True)
    # Deux colonnes : image à gauche, texte à droite
    col1, col2 = st.columns([1, 2])  # [1,2] = image prend 1/3, texte 2/3
    
    with col1:
        st.image(logo, width=150)  # Ajuste la taille si besoin

    with col2:
        st.markdown("""
        ### Welcome to HOMER  
        <br>
        <div style="display: flex; align-items: center; gap: 15px;">
            <span>Start by uploading your documents</span>
            <a href="./Documents" target="_self">
                <button style="padding:0.3em 0.8em; font-size:16px; background-color:#984a9c; color:white; border:none; border-radius:5px; cursor:pointer;">
                    HERE
                </button>
            </a>
        </div>
        
        <br>
        Then, you can either ask a simple question or generate a full structured report based on the content of your files.
        """, unsafe_allow_html=True)
    
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
    background-color: #f4f3ed;
    padding: 10px 0;
    border-top: 1px solid #ddd;
    z-index: 100;
}
</style>

<div class="footer">
    Designed by <strong>Florent Bergé & Mathieu de la Barre</strong>
</div>
""", unsafe_allow_html=True)
    



if __name__ == "__main__":
    _init()
    main()