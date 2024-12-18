import requests
import csv
import time
import random
from datetime import datetime, timezone

# Function to fetch reviews for a single game

def fetch_reviews(app_id, cursor='*', num_reviews=100):
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    params = {
        'filter': 'recent',
        'language': 'english',
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page': num_reviews,
        'cursor': cursor
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch reviews for App ID {app_id}: {response.status_code}")
        return None

# Function to save reviews to a CSV file

def save_reviews_to_csv(reviews, file_name, app_id, exclude_phrase=None):
    with open(file_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for review in reviews:
            # Get the review text
            review_text = review.get('review', '')

            # Skip reviews containing the exclude_phrase
            if exclude_phrase and exclude_phrase.lower() in review_text.lower():
                continue  # Skip this review

            # Convert Unix timestamp to a timezone-aware readable format
            timestamp = review.get('timestamp_created', 0)
            readable_date = (
                datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S') 
                if timestamp else ''
            )

            # Process playtime_forever converted to hours safely as a float
            playtime_forever_raw = review.get('author', {}).get('playtime_forever', 0)
            playtime_forever = round(float(playtime_forever_raw / 60), 2) if playtime_forever_raw else 0

            # Process weighted_vote_score safely as a float
            weighted_vote_score_raw = review.get('weighted_vote_score', 0)
            weighted_vote_score = round(float(weighted_vote_score_raw), 2) if weighted_vote_score_raw else 0

            
            writer.writerow([
                review.get('recommendationid', ''),
                app_id,
                review.get('review', ''),
                readable_date,  # Use the converted timestamp
                playtime_forever, # Use the converted playtime in hours
                review.get('voted_up', ''),
                review.get('votes_up', 0),
                review.get('votes_funny', 0),
                weighted_vote_score, # Add the processed weighted_vote_score
                review.get('received_for_free', ''),
                review.get('written_during_early_access', ''),
                (''),
            ])

# Main function

def scrape_random_reviews(app_ids_file, output_csv, max_reviews_per_game=100, fetch_pool_size=1000):

    # Open the file containing App IDs
    with open(app_ids_file, 'r') as f:
        app_ids = [line.strip() for line in f.readlines()]

    # Write CSV headers
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Review ID',
            'Game ID', 
            'Review Text', 
            'Last Updated', 
            'Hours played', 
            'Recommended?', 
            'Votes Up', 
            'Votes Funny', 
            'Weighted Helpfulness Score', 
            'Received for free?', 
            'Early Access review?',
            'Helpful Yes/No? (Personal Evaluation)'
        ])

    # Iterate over each App ID and fetch reviews
    for app_id in app_ids:
        print(f"Fetching up to {fetch_pool_size} reviews for App ID {app_id}...")
        cursor = '*'
        all_reviews = []

        while len(all_reviews) < fetch_pool_size:
            data = fetch_reviews(app_id, cursor, num_reviews=100)  # Fetch 100 reviews at a time
            if data and 'reviews' in data:
                reviews = data['reviews']
                if not reviews:
                    break  # No more reviews to fetch

                all_reviews.extend(reviews)
                cursor = data['cursor']
                print(f"Fetched {len(reviews)} reviews so far for App ID {app_id}.")
                time.sleep(1)  # To avoid hitting API rate limits
            else:
                break

        # Limit the pool size to fetch_pool_size
        all_reviews = all_reviews[:fetch_pool_size]

        # Shuffle and randomly select max_reviews_per_game
        random.shuffle(all_reviews)
        selected_reviews = all_reviews[:max_reviews_per_game]
        save_reviews_to_csv(selected_reviews, output_csv, app_id, exclude_phrase="You forget what reality is")

    print(f"All reviews have been saved to {output_csv}.")

# Usage

if __name__ == "__main__":
    app_ids_file = "app_ids.txt"  # File containing App IDs (one per line)
    output_csv = "steam_reviews.csv"  # Output CSV file
    scrape_random_reviews(app_ids_file, output_csv, max_reviews_per_game=100)
