import os
import openai
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API client setup
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

class SentimentAnalyzer:
    def __init__(self):
        self.model = "llama3-70b-8192"  # Or "mixtral-8x7b-32768"

    def analyze_sentiment_batch(self, comments, batch_size=10):
        """Analyze sentiment for multiple comments in a single API call"""
        # For large batches, process in smaller chunks to avoid token limits
        if len(comments) > batch_size:
            all_results = []
            for i in range(0, len(comments), batch_size):
                batch = comments[i : i + batch_size]
                results = self.analyze_sentiment_batch(batch)
                all_results.extend(results)
            return all_results

        # For small batches, process directly
        combined_text = "\n---\n".join(comments)

        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You will analyze the sentiment of each comment. For each comment, respond with exactly one of these labels: NEGATIVE, NEUTRAL, or POSITIVE. Return just a comma-separated list with no other text.",
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of each comment and return only the sentiment labels as a comma-separated list:\n{combined_text}",
                    },
                ],
                max_tokens=100,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()

            # Extract just the sentiment values, handling various formats
            # Look for common patterns in the response
            if ":" in result_text or "\n" in result_text:
                # Extract the actual values, removing any explanatory text
                lines = result_text.split("\n")
                values = []

                for line in lines:
                    if any(
                        label in line.upper()
                        for label in ["NEGATIVE", "NEUTRAL", "POSITIVE"]
                    ):
                        # Extract the sentiment label from the line
                        if "NEGATIVE" in line.upper():
                            values.append("NEGATIVE")
                        elif "NEUTRAL" in line.upper():
                            values.append("NEUTRAL")
                        elif "POSITIVE" in line.upper():
                            values.append("POSITIVE")
            else:
                # Assume it's a comma-separated list
                values = [v.strip().upper() for v in result_text.split(",")]

            # If we found fewer values than comments, fill in the rest as NEUTRAL
            if len(values) < len(comments):
                values.extend(["NEUTRAL"] * (len(comments) - len(values)))

            # Ensure we only return valid sentiment labels
            valid_sentiments = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            sanitized_values = []

            for value in values:
                if any(sentiment in value for sentiment in valid_sentiments):
                    for sentiment in valid_sentiments:
                        if sentiment in value:
                            sanitized_values.append(sentiment)
                            break
                else:
                    sanitized_values.append("NEUTRAL")  # Default to NEUTRAL

            return sanitized_values[
                : len(comments)
            ]  # Only return as many results as we have comments

        except Exception as e:
            print(f"Batch sentiment analysis error: {e}")
            return ["NEUTRAL"] * len(comments)  # Fallback to neutral


if __name__ == "__main__":
    results = analyze_comments()

    if results:
        sentiments = [r['sentiment'] for r in results]
        total = len(sentiments)
        sentiment_counts = {
            'Positive': sentiments.count('Positive') / total * 100,
            'Neutral': sentiments.count('Neutral') / total * 100,
            'Negative': sentiments.count('Negative') / total * 100
        }

        print("\nOverall Sentiment Analysis:")
        for sentiment, percentage in sentiment_counts.items():
            print(f"{sentiment}: {percentage:.1f}%")