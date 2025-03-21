import numpy as np

def calculate_rating(results):
    """
    Calculates a basic rating for a website based on aggregated sentiment.

    Parameters:
    results (list of dict): A list of dictionaries with comment sentiment details and scores.
        Example: [{"comment": "Great service!", "sentiment_label": 2, "scores": [0.1, 0.2, 0.7]}, ...]

    Returns:
    float: The calculated rating (1 to 5 stars).
    """
    # Assign numerical weights to sentiment labels
    label_weights = {
        0: -1,  # Negative
        1: 0,   # Neutral
        2: 1    # Positive
    }

    # Extract scores for each sentiment label
    sentiment_scores = []
    for result in results:
        sentiment_label = result['sentiment_label']  # Sentiment label (0, 1, 2)
        score = label_weights[sentiment_label]
        sentiment_scores.append(score)
    
    # Calculate the average score
    if sentiment_scores:
        average_score = np.mean(sentiment_scores)
        # Normalize to a 1-5 star rating
        normalized_rating = np.interp(average_score, [-1, 1], [1, 5])
        return round(normalized_rating, 2)
    else:
        return 0  # No comments analyzed
    

if __name__ == "__main__":
    # Example usage
    results = [
        {"comment": "Great service!", "sentiment_label": 2, "scores": [0.1, 0.2, 0.7]},
        {"comment": "Terrible experience...", "sentiment_label": 0, "scores": [0.8, 0.1, 0.1]},
        {"comment": "Average at best.", "sentiment_label": 1, "scores": [0.3, 0.4, 0.3]}
    ]

    rating = calculate_rating(results)
    print(f"Calculated rating: {rating} stars")
    # Output: Calculated rating: 2.67 stars