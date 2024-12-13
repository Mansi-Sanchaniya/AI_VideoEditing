import streamlit as st
import yt_dlp
from pytube import Playlist
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
import os
import time

# Helper Functions

def get_video_urls(input_urls):
    video_urls = []
    urls = input_urls.split(",")
    for url in urls:
        url = url.strip()
        if "playlist" in url:
            playlist = Playlist(url)
            video_urls.extend(playlist.video_urls)
        else:
            video_urls.append(url)
    return video_urls

def download_video(video_url, output_dir="downloads"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            return os.path.join(output_dir, f"{info['title']}.{info['ext']}")
    except Exception as e:
        st.error(f"Failed to download {video_url}: {e}")
        return None

def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except:
        return None

def format_transcript(transcript):
    return [f"[{entry['start']}s - {entry['start'] + entry['duration']}s] {entry['text']}" for entry in transcript]

def process_input(input_urls):
    video_urls = get_video_urls(input_urls)
    transcripts = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_transcript, url): url for url in video_urls}
        for future in as_completed(futures):
            url = futures[future]
            transcript = future.result()
            if transcript:
                formatted = format_transcript(transcript)
                transcripts.append({"video_url": url, "transcript": formatted})
            else:
                transcripts.append({"video_url": url, "transcript": ["Transcript not available"]})
    return transcripts

def process_query(query, transcripts, threshold=0.3):
    corpus = []
    metadata = []

    for video in transcripts:
        for line in video['transcript']:
            corpus.append(line)
            metadata.append(video['video_url'])

    vectorizer = TfidfVectorizer(stop_words='english')
    text_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, text_vectors).flatten()
    results = [
        {"text": corpus[i], "video_url": metadata[i], "index": i, "score": scores[i]}
        for i, score in enumerate(scores) if score > threshold
    ]

    return results

def extract_clips(results, transcripts, output_dir="clips"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clips = []
    for result in results:
        video_url = result["video_url"]
        video_path = download_video(video_url)
        if not video_path:
            continue
        
        transcript_index = result["index"]
        start_time, end_time = parse_time_range(transcripts, transcript_index)
        clip = VideoFileClip(video_path).subclip(start_time, end_time)
        clips.append(clip)

    return clips

def parse_time_range(transcripts, index):
    # Assuming `index` aligns with the start time and duration in transcripts
    for video in transcripts:
        for i, line in enumerate(video['transcript']):
            if i == index:
                start = float(line['start'])
                end = start + float(line['duration'])
                return start, end
    return 0, 0

def combine_clips(clips, output_file="combined_video.mp4"):
    if not clips:
        return None

    combined = concatenate_videoclips(clips, method="compose")
    combined.write_videofile(output_file, codec="libx264")
    return output_file

# Streamlit App

def main():
    st.set_page_config(page_title="YouTube Video Processor", page_icon="🎥", layout="wide")

    st.title("🎥 YouTube Video and Playlist Processor")
    input_urls = st.text_input("Enter YouTube Playlist or Video URLs (comma-separated):")

    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []

    if st.button("Extract Transcripts"):
        with st.spinner("Extracting transcripts, please wait..."):
            st.session_state.transcripts = process_input(input_urls)
            st.success("Transcripts extracted successfully!")

    if st.session_state.transcripts:
        st.subheader("Extracted Transcripts")
        for video in st.session_state.transcripts:
            st.write(f"**{video['video_url']}**")
            st.text("\n".join(video['transcript']))

    query = st.text_input("Enter your query:")

    if query and st.button("Search Transcripts"):
        with st.spinner("Searching transcripts, please wait..."):
            results = process_query(query, st.session_state.transcripts)
            if results:
                st.subheader("Query Results")
                st.write("\n".join([res["text"] for res in results]))

                if st.button("Combine and Play"):
                    with st.spinner("Processing clips..."):
                        clips = extract_clips(results, st.session_state.transcripts)
                        output_file = combine_clips(clips)
                        if output_file:
                            st.video(output_file)
                        else:
                            st.error("Failed to combine clips.")
            else:
                st.warning("No relevant content found.")

if __name__ == "__main__":
    main()
