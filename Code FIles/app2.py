import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import validators
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# Load Groq API Key
api_key = os.getenv('GROQ_API_KEY')

# Streamlit page setup
## what you see at the top of Chrome
st.set_page_config(page_title="YT/Web Summarizer", page_icon="🦜")
st.title("🦜 LangChain: Summarize YT Videos or Websites")

# Sidebar for API key
with st.sidebar:
    api_key = st.text_input("GROQ API KEY", value=api_key or "", type="password")

# URL input
generic_url = st.text_input("Paste a YouTube or Website URL here", label_visibility="collapsed")

# Load Groq LLM
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)

# Prompt template
prompt_template = """
Provide a summary of the content in 300 words:
Context: {text}
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

# Function to get transcript from YouTube
# Extracts the video ID from the YouTube URL.
# Uses YouTubeTranscriptApi to fetch the transcript text.
# Combines the transcript chunks into one big string.
def get_youtube_transcript(youtube_url):
    video_id = parse_qs(urlparse(youtube_url).query).get("v")
    if not video_id:
        return None
    video_id = video_id[0]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        st.error(f"❌ Transcript error: {e}")
        return None

# Button click logic
if st.button("Summarize the content from YT or Website"):
    if not api_key.strip() or not generic_url.strip():
        st.error("Please provide both API key and a URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("⏳ Summarizing..."):
            try:
                if "youtube.com" in generic_url:
                    transcript_text = get_youtube_transcript(generic_url)
                    if transcript_text:
                        docs = [Document(page_content=transcript_text)]
                    else:
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"user_agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                result = chain.run(docs)

                st.success("✅ Summary:")
                st.write(result)
            except Exception as e:
                st.error(f"🚨 Exception: {e}")
