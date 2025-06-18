import streamlit as st

st.set_page_config(page_title="Report", page_icon="🔍")

st.markdown("# Query")
st.sidebar.header("Query")

def main():
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create the query input area at the bottom
    query = st.chat_input("Enter your query:")

    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        history.append(query)

        # Display the user message immediately
        with st.chat_message

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.error("You are not logged in")
    elif not st.session_state.logged_in:
        st.error("You are not logged in")
    else:
        main()