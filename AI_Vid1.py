import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# Function to get the transcript for a YouTube video
def get_transcript(video_url):
    # Extract video ID from URL
    video_id = video_url.split("v=")[-1]
    
    try:
        # Try to fetch the transcript for the given video ID
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except NoTranscriptFound:
        # If no transcript is found for the video
        return f"Transcript not available for video: {video_url}."
    except TranscriptsDisabled:
        # If transcripts are disabled for the video
        return f"Transcripts are disabled for video: {video_url}."
    except Exception as e:
        # Any other exceptions (e.g., network issues)
        return f"Error fetching transcript for video {video_url}: {str(e)}"

# Streamlit UI
st.title("YouTube Transcript Fetcher")

# User input for the video URL
user_input = st.text_input("Enter YouTube Video URL:")

if st.button("Get Transcript"):
    if user_input:
        st.write(f"Fetching transcript for video: {user_input}")
        
        # Get the transcript
        result = get_transcript(user_input)
        
        # If the result is a list (transcript), display it
        if isinstance(result, list):
            formatted_transcript = [f"{entry['start']} - {entry['start'] + entry['duration']}: {entry['text']}" for entry in result]
            st.text_area("Transcript", "\n".join(formatted_transcript), height=300)
        else:
            # If the result is an error message, display it directly
            st.error(result)
    else:
        st.warning("Please enter a valid YouTube video URL.")
