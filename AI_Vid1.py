import streamlit as st
import yt_dlp
from pytube import Playlist
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy import VideoFileClip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.video.compositing.concatenate import concatenate_videoclips
import os
import time
import io
import yt_dlp
import subprocess
import logging

# Configure logging for better visibility in production
logging.basicConfig(level=logging.INFO)
from pytube import Playlist
import logging
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    VideoUnavailable,
    NoTranscriptFound
)

def get_video_urls_multiple(input_urls):
    video_urls = []
    urls = input_urls.split(",")  # Split input by comma
    for url in urls:
        url = url.strip()  # Remove any leading/trailing spaces
        try:
            if "playlist" in url:
                playlist = Playlist(url)
                video_urls.extend(playlist.video_urls)  # Add all video URLs in the playlist
            else:
                video_urls.append(url)  # Treat as a single video URL
        except Exception as e:
            # Log or handle errors to ensure deployment doesn't fail
            print(f"Error processing URL {url}: {e}")
    return video_urls


def download_video(video_url):
    """
    Downloads a video using yt_dlp and creates a Streamlit download button for it.
    """
    try:
        buffer = io.BytesIO()  # Create an in-memory bytes buffer for the video file
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',  # Download best video and audio
            'quiet': True,  # Suppress yt-dlp logs
            'outtmpl': '-',  # Output to stdout
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)  # Get metadata
            ydl.download([video_url])  # Download the video

            filename = ydl.prepare_filename(info_dict)
            with open(filename, 'rb') as f:
                buffer.write(f.read())  # Read the downloaded video into memory
                buffer.seek(0)  # Reset the buffer position

        # Provide the video as a download button in Streamlit
        st.download_button(
            label="Download Video",
            data=buffer,
            file_name=f"{info_dict['title']}.mp4",
            mime="video/mp4"
        )
        return filename

    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None

def get_transcript(video_url):
    """
    Fetches the transcript of a YouTube video using its URL.
    """
    try:
        # Extract the video ID from the URL
        video_id = video_url.split("v=")[-1]
        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except TranscriptsDisabled:
        logging.warning(f"Transcripts are disabled for video: {video_url}")
        return None
    except NoTranscriptFound:
        logging.warning(f"No transcript found for video: {video_url}")
        return None
    except VideoUnavailable:
        logging.error(f"Video unavailable: {video_url}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

def format_transcript(transcript):
    formatted_transcript = []
    for entry in transcript:
        try:
            start_time = entry['start']  # Timestamp
            duration = entry['duration']
            text = entry['text']  # Transcript text
            
            if not isinstance(start_time, (int, float)) or not isinstance(duration, (int, float)):
                raise ValueError(f"Invalid data types: start_time ({start_time}) or duration ({duration})")
            
            if not isinstance(text, str):
                raise ValueError(f"Invalid text value: {text}")

            # Append formatted string
            formatted_transcript.append(f"[{start_time}s - {start_time + duration}s] {text}")
        except KeyError as e:
            formatted_transcript.append(f"Missing key: {e}")
        except ValueError as e:
            formatted_transcript.append(str(e))

    return formatted_transcript




# Assuming you have the following helper functions
# get_video_urls_multiple(input_urls), get_transcript(video_url), format_transcript(transcript)

def process_input(input_urls):
    video_urls = get_video_urls_multiple(input_urls)
    
    if not video_urls:
        return []

    all_transcripts = []  # List to hold all transcripts
    video_chunks = {}  # Dictionary to store video-specific transcripts

    # Use ThreadPoolExecutor to fetch transcripts concurrently
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
                st.error(f"Error getting transcript for {video_url}: {e}")

    # Reassemble the output in the original order of video URLs
    for video_url in video_urls:
        all_transcripts.append(
            {"video_url": video_url, "transcript": video_chunks.get(video_url, ["No transcript found"])}
        )
    return all_transcripts


def process_query(query, stored_transcripts, threshold=0.3):  
    if not query:
        st.warning("Please enter a query to search in the transcripts.")
        return []

    if not stored_transcripts:
        st.warning("No transcripts available. Please process a playlist or video first.")
        return []

    all_transcripts_text = []
    video_info_list = []

    # Process all video transcripts
    for video in stored_transcripts:
        video_url = video.get('video_url', 'Unknown Video')
        transcript = video.get('transcript', [])
        
        if isinstance(transcript, list):
            for line in transcript:
                all_transcripts_text.append(line)
                video_info_list.append(f"Video: {video_url}\n")
        else:
            st.warning(f"Invalid transcript format for video: {video_url}. Skipping this video.")
    
    # Check if any valid transcripts are found
    if not all_transcripts_text:
        st.warning("No valid transcripts found in the stored data.")
        return []

    # Initialize TF-IDF Vectorizer and calculate similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = all_transcripts_text
    try:
        query_vector = vectorizer.fit_transform([query])
        text_vectors = vectorizer.transform(corpus)
    except ValueError as e:
        st.error(f"Error during vectorization: {e}")
        return []

    cosine_similarities = cosine_similarity(query_vector, text_vectors)

    # Find and return relevant sections above the threshold
    relevant_sections = []
    for idx, score in enumerate(cosine_similarities[0]):
        if score > threshold:  # Only include sections that pass the similarity threshold
            relevant_sections.append(f"{video_info_list[idx]}{all_transcripts_text[idx]}")

    if not relevant_sections:
        st.warning("No relevant sections found that match your query with the given threshold.")
    
    return relevant_sections

def edit_video(video_file, relevant_sections, ffmpeg_location):
    if not relevant_sections:
        logging.warning("No relevant sections provided.")
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
            else:
                logging.warning("Section processing failed.")

    if clips:
        # Save the final video to a temporary location
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                final_video_path = temp_file.name
            logging.info(f"Merging clips into {final_video_path}")
            merge_clips_with_ffmpeg(clips, final_video_path, ffmpeg_location)
            logging.info("Video editing complete.")
            return final_video_path
        except Exception as e:
            logging.error(f"Error while merging clips: {e}")
            return None
    else:
        logging.warning("No clips were generated.")
        return None

def process_section(section, video_file, ffmpeg_location):
    # Process a section of the video
    try:
        # Implement section processing logic here, such as extracting clips, trimming, etc.
        # Example:
        # clip = extract_clip(section, video_file, ffmpeg_location)
        # return clip
        pass
    except Exception as e:
        logging.error(f"Error processing section {section}: {e}")
        return None

def merge_clips_with_ffmpeg(clips, output_path, ffmpeg_location):
    # Assuming clips is a list of file paths of the individual clips
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            # Write a file list for FFmpeg
            for clip in clips:
                temp_file.write(f"file '{clip}'\n")

            temp_file_path = temp_file.name

        # Run FFmpeg command to merge clips
        ffmpeg_command = [ffmpeg_location, '-f', 'concat', '-safe', '0', '-i', temp_file_path, '-c', 'copy', output_path]
        subprocess.run(ffmpeg_command, check=True)
        os.remove(temp_file_path)  # Clean up temp file list
    except Exception as e:
        logging.error(f"Error during FFmpeg merge: {e}")
        raise


def process_section(section, video_file, ffmpeg_location):
    try:
        # Extract timestamps from section
        timestamps = extract_timestamps_from_section(section)
        if timestamps is None:
            logging.warning(f"Invalid timestamps for section: {section}. Skipping.")
            return None
        
        start_time, end_time = timestamps

        # Validate that the timestamps are reasonable (e.g., start < end and within the video length)
        if start_time >= end_time:
            logging.warning(f"Invalid time range for section: {section}. Start time is greater than end time.")
            return None
        
        # Extract clip using FFmpeg
        clip_segment = extract_clip_with_ffmpeg(video_file, start_time, end_time, ffmpeg_location)
        
        # Check if clip extraction succeeded
        if clip_segment is None:
            logging.error(f"Failed to extract clip for section: {section}.")
            return None

        return clip_segment

    except Exception as e:
        logging.error(f"Error processing section '{section}': {e}")
        return None

def extract_timestamps_from_section(section):
    # Dummy function for timestamp extraction; implement actual logic
    try:
        # Example: If section has the format 'start_time-end_time'
        times = section.split('-')
        if len(times) != 2:
            return None
        start_time, end_time = float(times[0]), float(times[1])
        return start_time, end_time
    except Exception as e:
        logging.error(f"Error extracting timestamps from section: {section} - {e}")
        return None

def extract_clip_with_ffmpeg(video_file, start_time, end_time, ffmpeg_location):
    try:
        # Ensure FFmpeg is accessible
        if not os.path.exists(ffmpeg_location):
            logging.error(f"FFmpeg not found at {ffmpeg_location}")
            return None

        # Create a temporary file for the clip
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            output_clip = temp_file.name

        # FFmpeg command to extract the clip
        ffmpeg_command = [
            ffmpeg_location, 
            '-i', video_file, 
            '-ss', str(start_time), 
            '-to', str(end_time), 
            '-c', 'copy', 
            output_clip
        ]
        
        # Run FFmpeg command
        subprocess.run(ffmpeg_command, check=True)

        return output_clip  # Return the path to the extracted clip

    except Exception as e:
        logging.error(f"Error extracting clip with FFmpeg: {e}")
        return None


def extract_timestamps_from_section(section):
    try:
        # Strip any leading/trailing whitespaces from the section
        section = section.strip()

        # Ensure the section contains timestamp information in the expected format
        if '[' not in section or ']' not in section:
            logging.warning(f"Section does not contain valid timestamps: {section}")
            return None

        # Extract the timestamp part from the section (inside the brackets)
        timestamp_part = section[section.find('[') + 1:section.find(']')].strip()
        times = timestamp_part.split(" - ")

        # Check if exactly two timestamps are found
        if len(times) != 2:
            logging.warning(f"Invalid timestamp format in section: {section}")
            return None

        # Clean and convert timestamps to floats, removing unnecessary "s"
        start_time = float(times[0].strip().replace("s", ""))
        end_time = float(times[1].strip().replace("s", ""))

        # Validate that the start time is less than the end time
        if start_time >= end_time:
            logging.warning(f"Invalid timestamp range in section: {section}. Start time is greater than or equal to end time.")
            return None

        # Round to a reasonable precision
        start_time = round(start_time, 2)
        end_time = round(end_time, 2)

        return start_time, end_time

    except Exception as e:
        logging.error(f"Error extracting timestamps from section '{section}': {e}")
        return None


def extract_clip_with_ffmpeg(video_file, start_time, end_time, ffmpeg_location):
    try:
        temp_output = f"temp_{start_time}_{end_time}.mp4"
        
        # Extract both video and audio while ensuring proper sync
        result = subprocess.run(
            [ffmpeg_location, "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
             "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-async", "1", temp_output],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if FFmpeg ran successfully
        if result.returncode != 0:
            logging.error(f"FFmpeg error for video {video_file}: {result.stderr.decode('utf-8')}")
            return None
        
        return temp_output

    except Exception as e:
        logging.error(f"Error extracting clip from video {video_file} between {start_time}s and {end_time}s: {e}")
        return None
    finally:
        # Cleanup temp files if necessary in future
        if os.path.exists(temp_output):
            os.remove(temp_output)


def merge_clips_with_ffmpeg(clips, output_file, ffmpeg_location):
    temp_clips_file = "temp_clips.txt"
    
    try:
        # Write the list of clips to a temporary text file
        with open(temp_clips_file, "w") as file:
            for clip in clips:
                file.write(f"file '{clip}'\n")

        # Merge clips using FFmpeg and ensure that audio-video sync is preserved
        result = subprocess.run([ffmpeg_location, "-f", "concat", "-safe", "0", "-i", temp_clips_file,
                                 "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-async", "1", output_file],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if FFmpeg ran successfully
        if result.returncode != 0:
            logging.error(f"FFmpeg error merging clips: {result.stderr.decode('utf-8')}")
            return None
        
        logging.info(f"Successfully merged clips into {output_file}")
        return output_file

    except Exception as e:
        logging.error(f"Error merging clips: {e}")
        return None

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_clips_file):
            os.remove(temp_clips_file)


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

def process_transcripts(input_urls, progress_bar, status_text):
    try:
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

    except Exception as e:
        logging.error(f"Error in process_transcripts: {e}")
        st.error(f"Error during transcript extraction: {e}")
        return None


# Function to simulate video processing
def process_video(query, progress_bar, status_text):
    try:
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

    except Exception as e:
        logging.error(f"Error in process_video: {e}")
        st.error(f"Error during video processing: {e}")
        return None


# Function to simulate combining and playing video
def combine_and_play_video(input_urls, progress_bar, status_text):
    try:
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

    except Exception as e:
        logging.error(f"Error in combine_and_play_video: {e}")
        st.error(f"Error during video combination: {e}")
        return None




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
                progress_bar = col2.progress(0, text="Starting transcript extraction. Please hold...")
                status_text = col2.empty()

                try:
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
                except Exception as e:
                    st.error(f"Error extracting transcripts: {e}")
                    logging.error(f"Error extracting transcripts: {e}")

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

                try:
                    relevant_sections = process_query(query, st.session_state.stored_transcripts)
                    progress_bar.progress(50, text="Analyzing query...")
                    status_text.text("Analyzing query...")
                    progress_bar.progress(100, text="Query processed successfully.")
                    status_text.text("Query processed successfully.")

                    if relevant_sections:
                        st.session_state.query_output = "\n".join(relevant_sections)
                    else:
                        st.session_state.query_output = "No relevant content found for the query."
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    logging.error(f"Error processing query: {e}")

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

            try:
                for video in st.session_state.stored_transcripts:
                    video_url = video['video_url']
                    video_file = download_video(video_url)
                    if not video_file:
                        st.warning(f"Skipping video {video_url} due to download failure.")
                    else:
                        status_text.text("Downloading video...")
                        progress_bar.progress(50, text="Video downloaded successfully...")
                        status_text.text("Processing will take some time. Please be patient...")
                        st.success(f"Video downloaded successfully: {video_file}")
                        query_output = st.session_state.query_output.split("\n") if st.session_state.query_output else []
                        edited_video_path = edit_video_using_query(video_file, query_output, "ffmpeg")
                        if edited_video_path:
                            progress_bar.progress(100, text="Video combined and ready.")
                            status_text.text("Video combined and ready.")
                            st.success(f"Edited video saved at: {edited_video_path}")
                            print('Final Video Created')
                            st.video(edited_video_path)
                            st.session_state.processing_in_progress = False
                        else:
                            st.warning("No video could be edited from the query output.")
            except Exception as e:
                st.error(f"Error in video processing: {e}")
                logging.error(f"Error in video processing: {e}")

    if st.button("Process Another Playlist/Video"):
        st.session_state.stored_transcripts = []
        st.rerun()

if __name__ == "__main__":
    main()
