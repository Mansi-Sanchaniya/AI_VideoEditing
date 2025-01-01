import os
import tempfile
import shutil
from moviepy.editor import VideoFileClip, concatenate_videoclips
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api._errors import NoTranscriptFound
from yt_dlp import YoutubeDL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Streamlit Sidebar for Input
st.sidebar.title("YouTube Playlist Processor")
cookie_file = st.sidebar.file_uploader("Upload cookies.txt for private playlists (optional)")
input_link = st.sidebar.text_input("Enter YouTube Playlist/Video URL")
query = st.sidebar.text_input("Enter Query to Search in Transcripts")
process_button = st.sidebar.button("Process")

# Prepare temporary directories
if not os.path.exists("temp_videos"):
    os.makedirs("temp_videos")

# Convert cookies file to compatible format (if provided)
def prepare_cookies(file):
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_cookie:
            temp_cookie.write(file.read())
            return temp_cookie.name
    return None

# Extract video URLs from a playlist or single video
@st.cache_data
def get_video_urls(link, cookies):
    ydl_opts = {
        'quiet': True,
        'cookiefile': cookies,
        'extract_flat': True if 'playlist' in link else False
    }
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=False)
        if 'entries' in result:
            return [video['url'] for video in result['entries']]
        else:
            return [result['webpage_url']]

# Download videos
@st.cache_data
def download_video(url, cookies):
    temp_path = tempfile.mkdtemp(dir="temp_videos")
    ydl_opts = {
        'quiet': True,
        'cookiefile': cookies,
        'outtmpl': os.path.join(temp_path, '%(id)s.%(ext)s'),
        'format': 'mp4'
    }
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.download([url])
        return temp_path

# Extract transcripts
@st.cache_data
def extract_transcripts(video_ids):
    transcripts = {}
    for video_id in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcripts[video_id] = transcript
        except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound):
            transcripts[video_id] = []
    return transcripts

# Search transcripts with a query
def search_transcripts(transcripts, query):
    results = {}
    tfidf = TfidfVectorizer(stop_words='english')
    for video_id, transcript in transcripts.items():
        if not transcript:
            continue
        texts = [entry['text'] for entry in transcript]
        tfidf_matrix = tfidf.fit_transform(texts)
        query_vec = tfidf.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        relevant_indices = similarities.argsort()[-5:][::-1]  # Top 5 matches
        results[video_id] = [transcript[i] for i in relevant_indices]
    return results

# Generate clips based on timestamps
def generate_clips(video_path, matches):
    clips = []
    video = VideoFileClip(video_path)
    for match in matches:
        start_time = match['start']
        end_time = match['start'] + match['duration']
        clip = video.subclip(start_time, end_time)
        clips.append(clip)
    video.close()
    return clips

# Main Processing
if process_button:
    if not input_link:
        st.sidebar.error("Please provide a valid YouTube URL.")
    else:
        cookies_path = prepare_cookies(cookie_file)
        video_urls = get_video_urls(input_link, cookies_path)
        st.write(f"Found {len(video_urls)} videos. Processing...")

        # Process each video
        for url in video_urls:
            st.write(f"Processing: {url}")
            video_id = url.split("v=")[-1]

            # Download video
            video_path = download_video(url, cookies_path)
            video_file = [file for file in os.listdir(video_path) if file.endswith('.mp4')][0]
            full_video_path = os.path.join(video_path, video_file)

            # Extract transcript
            transcripts = extract_transcripts([video_id])

            # Query transcripts
            search_results = search_transcripts(transcripts, query)
            if video_id in search_results and search_results[video_id]:
                clips = generate_clips(full_video_path, search_results[video_id])

                # Concatenate clips and save
                if clips:
                    final_clip = concatenate_videoclips(clips)
                    output_path = f"output_{video_id}.mp4"
                    final_clip.write_videofile(output_path, codec="libx264")
                    st.video(output_path)
            else:
                st.write("No relevant sections found in the transcript.")

        st.sidebar.success("Processing completed!")

# Cleanup temp files
def cleanup():
    shutil.rmtree("temp_videos", ignore_errors=True)

def main():
    try:
        if process_button:
            cleanup()
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
