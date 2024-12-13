import streamlit as st
import yt_dlp
from pytube import Playlist, YouTube
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from googleapiclient.discovery import build
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

ffmpeg_path = 'C:\\Users\\Vaishali\\PycharmProjects\\AI\\ffmpeg\\bin\\ffmpeg.exe'

# Function to get video URLs from multiple playlists or individual video links
def get_video_urls_multiple(input_urls):
    video_urls = []
    urls = input_urls.split(",")  # Split input by comma
    for url in urls:
        url = url.strip()  # Remove any leading/trailing spaces
        if "playlist" in url:
            playlist = Playlist(url)
            video_urls.extend(playlist.video_urls)  # Add all video URLs in the playlist
        else:
            video_urls.append(url)  # Treat as a single video URL
    return video_urls


def download_video(video_url, output_dir="downloads"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_template = os.path.join(output_dir, '%(title)s.%(ext)s')

    ydl_opts = {
        'ffmpeg_location': ffmpeg_path,
        'outtmpl': output_file_template,
        'format': 'bestvideo+bestaudio/best',  # Ensure video and audio are downloaded together
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info_dict)

            # Ensure filename is valid
            if filename and os.path.exists(filename):
                return filename
            else:
                print(f"Failed to retrieve valid filename for {video_url}")
                return None
    except Exception as e:
        print(f"Download failed for {video_url}: {str(e)}")
        return None


# Function to get transcript for a video using its YouTube ID
def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        # Fetch the transcript (if available)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return None


# Function to check if a video is under Creative Commons license using YouTube Data API and description

# Function to format the transcript into a readable form
def format_transcript(transcript):
    formatted_transcript = []
    for entry in transcript:
        start_time = entry['start']  # Timestamp
        duration = entry['duration']
        text = entry['text']  # Transcript text
        formatted_transcript.append(f"[{start_time}s - {start_time + duration}s] {text}")
    return formatted_transcript


# Function to process input (multiple playlists or individual videos) and fetch transcripts for all videos
def process_input(input_urls):
    video_urls = get_video_urls_multiple(input_urls)
    if not video_urls:
        return []

    all_transcripts = []  # List to hold all transcripts

    video_chunks = {}  # Dictionary to store video-specific transcripts

    # Use another ThreadPoolExecutor to fetch transcripts concurrently
    with ThreadPoolExecutor() as transcript_executor:
        future_to_video = {transcript_executor.submit(get_transcript, video_url): video_url for video_url in video_urls}
        for idx, future in enumerate(as_completed(future_to_video)):
            video_url = future_to_video[future]
            try:
                transcript = future.result()
                if transcript:
                    formatted_transcript = format_transcript(transcript)
                    video_chunks[video_url] = formatted_transcript  # Store by video URL
                else:
                    video_chunks[video_url] = ["Transcript not available"]
            except Exception as e:
                video_chunks[video_url] = ["Transcript extraction failed"]
                print(f"Error getting transcript for {video_url}: {e}")

    # Reassemble the output in the original order of video URLs
    for video_url in video_urls:
        all_transcripts.append(
            {"video_url": video_url, "transcript": video_chunks.get(video_url, ["No transcript found"])})
    return all_transcripts

# Function to process the query and extract relevant transcript segments
def process_query(query, stored_transcripts, threshold=0.3):  # Adjusted threshold for more precise results
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

    # Calculate cosine similarity using sklearn's cosine_similarity (which works directly with sparse matrices)
    cosine_similarities = cosine_similarity(query_vector, text_vectors)

    # Now, cosine_similarities will be a 2D numpy array where we can access the first row (the result for the query)
    relevant_sections = []
    for idx, score in enumerate(cosine_similarities[0]):
        if score > threshold:  # Only include sections that pass the similarity threshold
            relevant_sections.append(corpus[idx])

    return relevant_sections




def edit_video(video_file, relevant_sections, ffmpeg_location):
    if not relevant_sections:
        return None

    clips = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for section in relevant_sections:
            futures.append(executor.submit(process_section, section, video_file, ffmpeg_location))

        # Collect results from futures as they complete
        for future in as_completed(futures):
            result = future.result()
            if result:
                clips.append(result)

    if clips:
        final_video_path = "edited_video.mp4"
        merge_clips_with_ffmpeg(clips, final_video_path, ffmpeg_location)
        return final_video_path
    else:
        return None

def process_section(section, video_file, ffmpeg_location):
    timestamps = extract_timestamps_from_section(section)
    if timestamps is None:
        return None

    start_time, end_time = timestamps
    clip_segment = extract_clip_with_ffmpeg(video_file, start_time, end_time, ffmpeg_location)
    return clip_segment

def extract_timestamps_from_section(section):
    try:
        # Strip any leading/trailing whitespaces
        section = section.strip()

        # Check if the section contains timestamp information in the correct format
        if '[' not in section or ']' not in section:
            return None  # Skip sections that do not contain timestamps in '[start_time - end_time]' format

        # Extract the timestamp part of the section (the part inside the brackets)
        timestamp_part = section[section.find('[') + 1:section.find(']')].strip()  # Extract content inside brackets
        times = timestamp_part.split(" - ")

        # Ensure two timestamps are found in the section
        if len(times) != 2:
            return None  # Return None to skip this section

        # Clean timestamps and remove any unnecessary decimal precision
        start_time = float(times[0].strip().replace("s", ""))
        end_time = float(times[1].strip().replace("s", ""))

        # Round to a reasonable precision (e.g., 2 decimal places)
        start_time = round(start_time, 2)
        end_time = round(end_time, 2)

        return start_time, end_time
    except Exception as e:
        print(f"Error extracting timestamps from section '{section}'. Exception: {e}")
        return None  # Return None in case of an error




def extract_clip_with_ffmpeg(video_file, start_time, end_time, ffmpeg_location):
    try:
        temp_output = f"temp_{start_time}_{end_time}.mp4"
        # Extract both video and audio while ensuring proper sync
        subprocess.run(
            [ffmpeg_location, "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
             "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-async", "1", temp_output],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return temp_output
    except Exception as e:
        print(f"Error extracting clip: {e}")
        return None


def merge_clips_with_ffmpeg(clips, output_file, ffmpeg_location):
    with open("temp_clips.txt", "w") as file:
        for clip in clips:
            file.write(f"file '{clip}'\n")

    # Merge clips and ensure that audio-video sync is preserved
    subprocess.run([ffmpeg_location, "-f", "concat", "-safe", "0", "-i", "temp_clips.txt",
                    "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-async", "1", output_file],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# Use timestamps to extract and combine video clips
def edit_video_using_query(video_file, query_output, ffmpeg_location):
    if not query_output:
        print("No query output to process for video editing.")
        return None

    try:
        clips = []
        print("Extracting clips...")

        with ThreadPoolExecutor() as executor:
            futures = []
            for section in query_output:
                futures.append(executor.submit(process_section, section, video_file, ffmpeg_location))

            # Collect results from futures as they complete
            for future in as_completed(futures):
                result = future.result()
                if result:
                    clips.append(result)

        if clips:
            final_video_path = "final_output_video.mp4"
            print("Merging clips...")
            merge_clips_with_ffmpeg(clips, final_video_path, ffmpeg_location)
            return final_video_path
        else:
            print("No valid clips were found.")
            return None
    except Exception as e:
        print(f"Error editing video: {str(e)}")
        return None


# Simulating your process functions for this demonstration
def process_transcripts(input_urls, progress_bar, status_text):
    total_steps = 100  # Example total steps for the process
    start_time = time.time()  # Track the start time
    for step in range(total_steps):
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        time_remaining = elapsed_time / (step + 1) * (total_steps - step - 1)  # Estimate remaining time
        time_remaining_str = f"{time_remaining:.2f} seconds remaining"  # Format remaining time

        time.sleep(0.1)  # Simulate a task
        progress_bar.progress(step + 1, text=f"Extracting transcripts: {step + 1}% done")
        status_text.text(time_remaining_str)  # Update the remaining time text

    return "Transcripts Extracted!"  # Once complete


def process_video(query, progress_bar, status_text):
    total_steps = 100  # Example total steps for the process
    start_time = time.time()  # Track the start time
    for step in range(total_steps):
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        time_remaining = elapsed_time / (step + 1) * (total_steps - step - 1)  # Estimate remaining time
        time_remaining_str = f"{time_remaining:.2f} seconds remaining"  # Format remaining time

        time.sleep(0.1)  # Simulate a task
        progress_bar.progress(step + 1, text=f"Processing video: {step + 1}% done")
        status_text.text(time_remaining_str)  # Update the remaining time text

    return "Video Processed!"  # Once complete


def combine_and_play_video(input_urls, progress_bar, status_text):
    total_steps = 100  # Example total steps for the process
    start_time = time.time()  # Track the start time
    for step in range(total_steps):
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        time_remaining = elapsed_time / (step + 1) * (total_steps - step - 1)  # Estimate remaining time
        time_remaining_str = f"{time_remaining:.2f} seconds remaining"  # Format remaining time

        time.sleep(0.1)  # Simulate a task
        progress_bar.progress(step + 1, text=f"Combining & playing video: {step + 1}% done")
        status_text.text(time_remaining_str)  # Update the remaining time text

    return "Video Combined & Played!"  # Once complete

# Streamlit UI with modern, formal layout
# Streamlit UI with modern, formal layout

def main():
    st.set_page_config(page_title="Video & Playlist Processor", page_icon="🎬", layout="wide")

    st.markdown("""
    <style>
        .css-1d391kg {padding: 30px;}
        .stTextArea>div>div>textarea {
            font-size: 14px;
            line-height: 1.8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #ff5c5c;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #ff7d7d;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Video and Playlist Processor")

    input_urls = st.text_input("Enter YouTube Playlist(s) or Video URL(s) or both (comma-separated):")

    if 'stored_transcripts' not in st.session_state:
        st.session_state.stored_transcripts = []
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = ""

    if input_urls:
        # Create columns for button and progress bar side by side
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("Extract Transcripts"):
                progress_bar = col2.progress(0, text="Starting transcript extraction Please Hold...")
                status_text = col2.empty()  # Placeholder for dynamic status updates

                st.session_state.stored_transcripts = process_input(input_urls)
                progress_bar.progress(50, text="Processing transcripts...")
                status_text.text("Processing transcripts...")
                progress_bar.progress(100, text="Transcripts extracted successfully.")
                status_text.text("Transcripts extracted successfully.")
                if st.session_state.stored_transcripts:
                    transcript_text = ""
                    for video in st.session_state.stored_transcripts:
                        transcript_text += f"\nTranscript for video {video['video_url']}:\n"
                        if isinstance(video['transcript'], list):
                            for line in video['transcript']:
                                transcript_text += line + "\n"
                        else:
                            transcript_text += video['transcript'] + "\n"
                        transcript_text += "-" * 50 + "\n"
                    st.session_state.transcript_text = transcript_text

    if st.session_state.transcript_text:
        st.subheader("Extracted Transcripts")
        st.text_area("Transcripts", st.session_state.transcript_text, height=300, key="transcripts_area")

    query = st.text_input("Enter your query to extract relevant information:")
    if query:
        # Create columns for button and progress bar side by side
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("Process Query"):
                progress_bar = col2.progress(0, text="Starting query processing...")
                status_text = col2.empty()

                relevant_sections = process_query(query, st.session_state.stored_transcripts)
                progress_bar.progress(50, text="Analyzing query...")
                status_text.text("Analyzing query...")
                progress_bar.progress(100, text="Query processed successfully.")
                status_text.text("Query processed successfully.")
                if relevant_sections:
                    st.session_state.query_output = "\n".join(relevant_sections)
                else:
                    st.session_state.query_output = "No relevant content found for the query."

    if 'query_output' in st.session_state and st.session_state.query_output:
        st.subheader("Relevant Output for Your Query")
        st.text_area("Query Output", st.session_state.query_output, height=300, key="query_output_area")

    # Update Streamlit UI button for combining video using query output
    # Create columns for button and progress bar side by side
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("Combine and Play Video"):
            progress_bar = col2.progress(0, text="Starting combine and play...")
            status_text = col2.empty()

            for video in st.session_state.stored_transcripts:
                video_url = video['video_url']
                video_file = download_video(video['video_url'])
                if not video_file:
                    st.warning(f"Skipping video {video_url} due to download failure.")
                else:
                    status_text.text("Downloading video...")
                    progress_bar.progress(50, text="Video downloaded successfully...")
                    status_text.text("Processing will take some time \n Please Have Patience...")
                    st.success(f"Video downloaded successfully: {video_file}")
                    query_output = st.session_state.query_output.split("\n") if st.session_state.query_output else []
                    edited_video_path = edit_video_using_query(video_file, query_output, ffmpeg_path)
                    if edited_video_path:
                        progress_bar.progress(100, text="Video combined and ready.")
                        status_text.text("Video combined and ready.")
                        st.success(f"Edited video saved at: {edited_video_path}")
                        print('Final Video Created')
                        st.video(edited_video_path)
                        st.session_state.processing_in_progress = False
                    else:
                        st.warning("No video could be edited from the query output.")

    if st.button("Process Another Playlist/Video"):
        st.session_state.stored_transcripts = []
        st.rerun()


if __name__ == "__main__":
    main()
