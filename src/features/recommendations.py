import pandas as pd

def get_creator_recommendations(df_metrics):
    """
    Analyzes video metrics and returns a list of actionable AI recommendations.
    
    Args:
        df_metrics: A dictionary or Series containing:
            - view_count
            - like_count
            - comment_count
            - channel_subscriber_count
    """
    recommendations = []
    
    views = df_metrics['view_count']
    likes = df_metrics['like_count']
    comments = df_metrics['comment_count']
    subs = df_metrics['channel_subscriber_count']
    
    # 1. Engagement Rate (Like/View Ratio)
    like_ratio = (likes / max(1, views)) * 100
    if like_ratio < 2.0:
        recommendations.append({
            "topic": "Engagement",
            "status": "Low",
            "advice": "Your Like-to-View ratio is below the 2% benchmark. Try adding a 'Like' call-to-action (CTA) in the first 30 seconds of your video to improve conversion."
        })
    elif like_ratio > 7.0:
        recommendations.append({
            "topic": "Engagement",
            "status": "High",
            "advice": "Excellent engagement! Your audience loves this format. Consider creating a 'Part 2' or a deep-dive on this specific topic."
        })
    
    # 2. Audience Interaction (Comment/View Ratio)
    comment_ratio = (comments / max(1, views)) * 100
    if comment_ratio < 0.1:
        recommendations.append({
            "topic": "Interaction",
            "status": "Low",
            "advice": "Comment volume is low. Try pinning a 'Question of the Day' as the first comment to spark a discussion among your viewers."
        })
    
    # 3. Reach/Fanbase Conversion
    view_sub_ratio = (views / max(1, subs)) * 100
    if view_sub_ratio < 5.0 and subs > 1000:
        recommendations.append({
            "topic": "Reach",
            "status": "Low",
            "advice": "Current views are low relative to your subscriber base. Check if your thumbnail and title are 'intriguing' enough to capture your existing fans' attention."
        })
    
    # 4. General Positive Reinforcement
    if not recommendations:
        recommendations.append({
            "topic": "Performance",
            "status": "Stable",
            "advice": "Your metrics are balanced and consistent. Maintain this upload frequency to build long-term authority."
        })
        
    return recommendations
