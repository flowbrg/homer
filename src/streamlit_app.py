import streamlit as st


############################## Page navigation ##############################


homePage = st.Page("./pages/home.py", title="Home", icon=":material/home:")
discussionPage = st.Page("./pages/discussion.py", title="Discussion", icon=":material/chat_bubble:")
reportPage = st.Page("./pages/report.py", title="Report", icon=":material/edit:")
documentsPage = st.Page("./pages/index.py", title="Documents", icon=":material/database_upload:")
configPage = st.Page("./pages/config.py", title="Configuration", icon=":material/settings:")
modelsPage = st.Page("./pages/models.py", title="Models", icon=":material/download:")

pg = st.navigation([
    homePage,
    discussionPage,
    reportPage,
    documentsPage,
    configPage,
    modelsPage
])

pg.run()