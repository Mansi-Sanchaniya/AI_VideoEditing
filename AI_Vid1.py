import streamlit as st
from pytube import YouTube, Playlist
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
import json

def get_transcript(video_id):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        return None

def query_transcripts(transcripts, query):
    context = "\n".join([f"[{t['start']:.2f}] {t['text']}" for t in transcripts])
    prompt = (
        f"You are an AI assistant. Use the following transcript to answer the query:\n\n"
        f"Transcript:\n{context}\n\nQuery: {query}\nAnswer with timestamps:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Answer queries based on context."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("YouTube Transcript Extractor & Query Processor")

st.header("1. Input YouTube Links and Cookies")

# YouTube Link Input
youtube_link = st.text_input("Enter a YouTube Video/Playlist Link:")

# Cookie File Input
cookie_file = st.file_uploader("Upload Your YouTube Cookie File:", type=["txt", "json"])

if cookie_file:
    cookies = json.load(cookie_file)
    st.success("Cookie file uploaded successfully!")
else:
    cookies = None

# Extract Transcripts
if st.button("Extract Transcripts"):
    if not youtube_link:
        st.error("Please provide a YouTube link.")
    elif not cookies:
        st.error("Please upload a cookie file.")
    else:
        try:
            transcripts = []
            if "playlist" in youtube_link:
                playlist = Playlist(youtube_link)
                video_ids = [YouTube(url).video_id for url in playlist.video_urls]
            else:
                video_ids = [YouTube(youtube_link).video_id]

            for video_id in video_ids:
                transcript = get_transcript(video_id)
                if transcript:
                    transcripts.extend(transcript)
            
            if transcripts:
                st.text_area("Extracted Transcripts:", value="\n".join([f"[{t['start']:.2f}] {t['text']}" for t in transcripts]), height=300)
                st.session_state.transcripts = transcripts
            else:
                st.error("No transcripts could be extracted.")
        except Exception as e:
            st.error(f"Error: {e}")

# Query Processing
st.header("2. Query the Transcripts")
query = st.text_input("Enter your query:")

if st.button("Process Query"):
    if "transcripts" not in st.session_state:
        st.error("Please extract transcripts first.")
    elif not query:
        st.error("Please provide a query.")
    else:
        try:
            answer = query_transcripts(st.session_state.transcripts, query)
            st.success("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")

# Add your OpenAI API key
openai.api_key = "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"
