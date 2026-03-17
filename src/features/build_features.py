import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_age_in_months(published_at_str, reference_date=None):
    """Calculates age in months to match the old dataset's 'video_month_old' logic."""
    if pd.isna(published_at_str):
        return 1
    try:
        published_date = pd.to_datetime(published_at_str).tz_localize(None)
        if reference_date is None:
            reference_date = datetime.utcnow()
        months = (reference_date.year - published_date.year) * 12 + reference_date.month - published_date.month
        return max(1, months)
    except:
        return 1

def engineer_features(df, is_training=False):
    """
    Engineers features to match the legacy dataset while using modern API outputs.
    Note: YouTube removed public dislikes, so we drop 'dislikeCount' from the legacy model.
    """
    df = df.copy()
    
    if is_training:
        # If training on the original dataset/data_final.csv
        features = pd.DataFrame()
        features['viewCount'] = df['viewCount']
        features['commentCount'] = df['commentCount']
        features['viewCount/video_month_old'] = df['viewCount/video_month_old']
        features['subscriberCount/videoCount'] = df['subscriberCount/videoCount']
        
        target = df['likeCount']
        return features.fillna(0), target
    else:
        # If doing live inference from raw get_data.py output
        features = pd.DataFrame()
        features['viewCount'] = df['view_count']
        features['commentCount'] = df['comment_count']
        
        current_time = datetime.utcnow()
        df['months_old'] = df['published_at'].apply(lambda x: calculate_age_in_months(x, current_time))
        features['viewCount/video_month_old'] = df['view_count'] / df['months_old']
        
        # Add 1 to avoid division by zero
        features['subscriberCount/videoCount'] = (df['channel_subscriber_count'] + 1) / (df['channel_video_count'] + 1)
        
        return features.fillna(0)
