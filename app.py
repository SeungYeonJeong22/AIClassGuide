import streamlit as st
import asyncio
from websocket_client import send_video

def main():
    # Initialize the session state for the checkbox
    if 'stop_checkbox' not in st.session_state:
        st.session_state.stop_checkbox = False

    # Start the video sending process
    asyncio.run(send_video())

if __name__ == "__main__":
    main()