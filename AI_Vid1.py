import streamlit as st
import yt_dlp
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import json
import time
from http.cookiejar import CookieJar
import urllib.request
import requests
from moviepy.editor import VideoFileClip, concatenate_videoclips


# Function to convert cookies into Netscape format for yt-dlp
def convert_to_netscape(cookies_file):
    if cookies_file.name.endswith('.json'):
        cookies_data = json.load(cookies_file)
        netscape_cookies = []
        for cookie in cookies_data:
            netscape_cookies.append(
                f"{cookie['domain']}\t{cookie['httpOnly']}\t{cookie['secure']}\t{cookie['expiry']}\t{cookie['name']}\t{cookie['value']}")
        netscape_filename = os.path.join(os.path.dirname(cookies_file.name), "cookies_netscape.txt")
        with open(netscape_filename, "w") as netscape_file:
            for cookie in netscape_cookies:
                netscape_file.write(f"{cookie}\n")
        return netscape_filename
    elif cookies_file.name.endswith('.txt'):
        return cookies_file.name
    else:
        raise ValueError("Unsupported cookie file format. Please upload a .json or .txt file.")

# Function to get video URLs from multiple playlists or individual video links
def get_video_urls_multiple(input_urls):
    video_urls = []
    urls = input_urls.split(",")
    for url in urls:
        url = url.strip()
        if "playlist" in url:
            try:
                playlist = Playlist(url)
                video_urls.extend(playlist.video_urls)
            except Exception as e:
                st.error(f"Error processing playlist URL: {url}. {e}")
        else:
            video_urls.append(url)
    return video_urls

# Download video using yt_dlp
def download_video(video_url, cookies_file, output_dir="downloads"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_template = os.path.join(output_dir, '%(title)s.%(ext)s')

    ydl_opts = {
        'outtmpl': output_file_template,
        'format': 'best',  # This ensures you get the best quality video and audio
        'quiet': True,
        'no_warnings': True,
        'cookiefile': cookies_file  # Use cookies if necessary
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info_dict)
            if filename and os.path.exists(filename):
                print(f"Downloaded video: {filename}")
                return filename
            else:
                print(f"Failed to download video: {video_url}")
                return None
    except Exception as e:
        st.error(f"Error downloading video: {video_url}. {e}")
        return None


# Get transcript for a video
def get_transcript(video_url, cookies_file):
    video_id = video_url.split("v=")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, cookies=cookies_file)
        return transcript
    except (NoTranscriptFound, TranscriptsDisabled):
        st.warning(f"Transcript not available for video: {video_url}.")
        return None
    except Exception as e:
        st.warning(f"Error fetching transcript for video: {video_url}. {e}")
        return None

# Format transcript
def format_transcript(transcript):
    formatted_transcript = []
    for entry in transcript:
        start_time = entry['start']
        duration = entry['duration']
        text = entry['text']
        formatted_transcript.append(f"[{start_time}s - {start_time + duration}s] {text}")
    return formatted_transcript

# Process input and fetch transcripts
def process_input(input_urls, cookies_file):
    video_urls = get_video_urls_multiple(input_urls)
    if not video_urls:
        st.warning("No valid video URLs provided.")
        return []

    all_transcripts = []
    video_chunks = {}

    with ThreadPoolExecutor(max_workers=10) as transcript_executor:
        future_to_video = {transcript_executor.submit(get_transcript, video_url, cookies_file): video_url for video_url in video_urls}
        for future in as_completed(future_to_video, timeout=timeout):
            video_url = future_to_video[future]
            try:
                transcript = future.result()
                if transcript:
                    formatted_transcript = format_transcript(transcript)
                    video_chunks[video_url] = formatted_transcript
                    print(f"Processed transcript for video: {video_url}")
                else:
                    video_chunks[video_url] = ["Transcript not available"]
            except TimeoutError:
                video_chunks[video_url] = ["Transcript extraction timed out"]
                st.warning(f"Timeout reached while processing video: {video_url}")
            except Exception as e:
                video_chunks[video_url] = ["Transcript extraction failed"]

    for video_url in video_urls:
        all_transcripts.append(
            {"video_url": video_url, "transcript": video_chunks.get(video_url, ["No transcript found"])})
    return all_transcripts

# Process query
def process_query(query, stored_transcripts, threshold=0.3):
    if not query:
        st.warning("Please enter a query to search in the transcripts.")
        return []

    if not stored_transcripts:
        st.warning("No transcripts available. Please process a playlist or video first.")
        return []

    all_transcripts_text = []
    for video in stored_transcripts:
        video_info = f"Video: {video['video_url']}\n"
        if isinstance(video['transcript'], list):
            for line in video['transcript']:
                all_transcripts_text.append(video_info + line)

    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = all_transcripts_text
    query_vector = vectorizer.fit_transform([query])
    text_vectors = vectorizer.transform(corpus)

    cosine_similarities = cosine_similarity(query_vector, text_vectors)

    relevant_sections = []
    for idx, score in enumerate(cosine_similarities[0]):
        if score > threshold:
            relevant_sections.append(corpus[idx])

    return relevant_sections


def extract_timestamps_from_section(section):
    try:
        section = section.strip()

        if '[' not in section or ']' not in section:
            return None

        timestamp_part = section[section.find('[') + 1:section.find(']')].strip()
        times = timestamp_part.split(" - ")

        if len(times) != 2:
            return None

        start_time = float(times[0].strip().replace("s", ""))
        end_time = float(times[1].strip().replace("s", ""))

        start_time = round(start_time, 2)
        end_time = round(end_time, 2)

        return start_time, end_time
    except Exception as e:
        return None



# Video editing using MoviePy
def edit_video(video_file, relevant_sections):
    if not relevant_sections:
        return None

    clips = []
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for clips: {temp_dir}")

    for section in relevant_sections:
        start_time, end_time = extract_timestamps_from_section(section)
        print(f"Start time: {start_time}, End time: {end_time}")
        if start_time and end_time:
            clip = extract_clip_with_moviepy(video_file, start_time, end_time, temp_dir)
            if clip:
                clips.append(clip)

    if clips:
        final_video_path = os.path.join(temp_dir, "edited_video.mp4")
        merge_clips_with_moviepy(clips, final_video_path)
        print(f"Final edited video saved at: {final_video_path}")
        return final_video_path
    else:
        return None

# Extract clip using MoviePy
def extract_clip_with_moviepy(video_file, start_time, end_time, temp_dir):
    try:
        video = VideoFileClip(video_file)
        clip = video.subclip(start_time, end_time)
        temp_output = os.path.join(temp_dir, f"temp_{start_time}_{end_time}.mp4")
        clip.write_videofile(temp_output, codec="libx264", audio_codec="aac", threads=4)
        print(f"Extracted clip from {start_time}s to {end_time}s: {temp_output}")
        return temp_output
    except Exception as e:
        return None

# Merge clips using MoviePy
def merge_clips_with_moviepy(clips, output_file):
    try:
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(output_file, codec="libx264", audio_codec="aac", threads=4)
        print(f"Merged clips into final video: {output_file}")
    except Exception as e:
        pass


# Process and display the results for each video
if __name__ == "__main__":
    st.title("Video Playlist Processor")

    st.sidebar.header("User Options")
    cookies_file = st.sidebar.file_uploader("Upload Your YouTube Cookies File", type=["txt", "json"])

    if cookies_file:
        cookies_path = convert_to_netscape(cookies_file)
        if cookies_path:
            st.success("Cookies file successfully converted to Netscape format.")

    user_input = st.text_area("Enter Playlist or Video URLs (comma-separated):")

    if st.button("Process Playlists"):
        if not cookies_file:
            st.warning("Please upload a cookies file.")
        else:
            transcripts = process_input(user_input, cookies_path)
            if transcripts:
                for transcript in transcripts:
                    st.write(f"Video: {transcript['video_url']}")
                    st.text_area("Transcript", "\n".join(transcript['transcript']), height=300, key=transcript['video_url'])

    query = st.text_input("Enter your query:")
    if st.button("Search Transcripts"):
        if not cookies_file:
            st.warning("Please upload a cookies file.")
        else:
            transcripts = process_input(user_input, cookies_path)
            if transcripts:
                results = process_query(query, transcripts)
                st.text_area("Query Results", "\n".join(results), height=300)


    if st.button("Combine and Create Video"):
        if not cookies_file:
            st.warning("Please upload a cookies file.")
        else:
            transcripts = process_input(user_input, cookies_path)
            relevant_sections = process_query(query, transcripts)
            if relevant_sections:
                video_path = download_video(user_input, cookies_path)  # Ensure the video is downloaded before editing
                if video_path:
                    edited_video_path = edit_video(video_path, relevant_sections)
                    if edited_video_path:
                        st.video(edited_video_path)
                    else:
                        st.error("Failed to create the video.")
                else:
                    st.error("Failed to download the video.")

