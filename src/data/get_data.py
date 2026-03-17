import os
import logging
from googleapiclient.discovery import build
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    logging.warning("YOUTUBE_API_KEY not found in environment variables. Data extraction will fail.")

def get_youtube_client():
    """Returns a built YouTube API client."""
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def fetch_video_data(youtube, video_ids):
    """Fetches statistics and snippet data for a list of video IDs."""
    if not video_ids:
        return []

    results = []
    # Can process up to 50 IDs at a time
    chunk_size = 50
    for i in range(0, len(video_ids), chunk_size):
        chunk = video_ids[i:i + chunk_size]
        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(chunk)
        )
        response = request.execute()
        results.extend(response.get('items', []))
    
    return results

def fetch_channel_data(youtube, channel_ids):
    """Fetches statistics and snippet data for a list of channel IDs."""
    if not channel_ids:
        return []

    unique_channel_ids = list(set(channel_ids))
    results = []
    chunk_size = 50
    for i in range(0, len(unique_channel_ids), chunk_size):
        chunk = unique_channel_ids[i:i + chunk_size]
        try:
            request = youtube.channels().list(
                part="snippet,statistics",
                id=",".join(chunk)
            )
            response = request.execute()
            results.extend(response.get('items', []))
        except Exception as e:
            logging.error(f"Error fetching channel data: {e}")
    
    return results

def get_full_video_stats(video_ids):
    """Orchestrates fetching video and their corresponding channel data."""
    youtube = get_youtube_client()
    
    logging.info(f"Fetching data for {len(video_ids)} videos...")
    video_data = fetch_video_data(youtube, video_ids)
    
    if not video_data:
        return []

    channel_ids = [item['snippet']['channelId'] for item in video_data if 'channelId' in item['snippet']]
    
    logging.info(f"Fetching data for {len(set(channel_ids))} channels...")
    channel_data = fetch_channel_data(youtube, channel_ids)
    
    # Create channel lookup mapping
    channel_lookup = {item['id']: item for item in channel_data}
    
    combined_data = []
    for video in video_data:
        v_stats = video.get('statistics', {})
        v_snippet = video.get('snippet', {})
        
        channel_id = v_snippet.get('channelId')
        channel_info = channel_lookup.get(channel_id, {})
        c_stats = channel_info.get('statistics', {})
        c_snippet = channel_info.get('snippet', {})
        
        # Build clean dictionary for DF
        row = {
            'video_id': video['id'],
            'title': v_snippet.get('title', ''),
            'channel_id': channel_id,
            'category_id': v_snippet.get('categoryId', ''),
            'published_at': v_snippet.get('publishedAt'),
            'view_count': int(v_stats.get('viewCount', 0)),
            'like_count': int(v_stats.get('likeCount', 0)),
            'comment_count': int(v_stats.get('commentCount', 0)),
            'description': v_snippet.get('description', ''),
            'channel_published_at': c_snippet.get('publishedAt'),
            'channel_view_count': int(c_stats.get('viewCount', 0)),
            'channel_subscriber_count': int(c_stats.get('subscriberCount', 0)),
            'channel_video_count': int(c_stats.get('videoCount', 0))
        }
        combined_data.append(row)
        
    return combined_data

if __name__ == "__main__":
    # Test execution
    test_ids = ["jNQXAC9IVRw", "dQw4w9WgXcQ"] # Me at the zoo, Rick Astley
    data = get_full_video_stats(test_ids)
    print(f"Fetched {len(data)} records successfully.")
