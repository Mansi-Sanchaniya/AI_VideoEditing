from youtube_transcript_api import YouTubeTranscriptApi

def test_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return str(e)

test_video_id = 'bixR-KIJKYM'  # Replace with a valid YouTube video ID
print(test_transcript(test_video_id))
