import re
from datetime import datetime

import pandas as pd
import wikipedia


def scrape_with_api():
    try:
        # Get the page content
        page = wikipedia.page("Timeline_of_the_20th_century")
        content = page.content
        
        # Parse the content for events
        lines = content.split('\n')
        events = []
        
        for line in lines:
            # Look for lines that start with dates
            if re.match(r'^\w+ \d{1,2}', line.strip()):
                events.append(line.strip())
        
        return events
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Multiple pages found: {e.options}")
    except wikipedia.exceptions.PageError:
        print("Page not found")

def scrape_20th_century_timeline():
    """
    Scrapes the Timeline of 20th Century Wikipedia page and returns a DataFrame
    with properly formatted dates and event text.
    """
    try:
        # Get the page content
        page = wikipedia.page("Timeline_of_the_20th_century")
        content = page.content
        
        # Parse the content for events
        lines = content.split('\n')
        events = []
        current_year = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this line contains a year (like "1900" or "== 1900 ==")
            year_match = re.search(r'(\d{4})', line)
            if year_match and len(line) < 20:  # Likely a year header
                current_year = year_match.group(1)
                continue
            
            # Look for lines with month/day format: "September 5: The event..."
            date_event_match = re.match(r'^([A-Za-z]+ \d{1,2}):\s*(.+)$', line)
            if date_event_match and current_year:
                month_day = date_event_match.group(1)
                event_text = date_event_match.group(2).strip()
                
                # Create full date string
                full_date = f"{month_day}, {current_year}"
                
                # Try to parse the date to ensure it's valid
                try:
                    parsed_date = datetime.strptime(full_date, "%B %d, %Y")
                    formatted_date = parsed_date.strftime("%Y-%m-%d")
                    
                    events.append({
                        'date': formatted_date,
                        'text': event_text,
                        'original_format': line
                    })
                except ValueError:
                    # If date parsing fails, still include with original format
                    events.append({
                        'date': full_date,
                        'text': event_text,
                        'original_format': line
                    })
        
        # Create DataFrame
        df = pd.DataFrame(events)
        return df
        
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Multiple pages found: {e.options}")
        return pd.DataFrame()
    except wikipedia.exceptions.PageError:
        print("Page not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error occurred: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("Scraping Wikipedia Timeline of 20th Century using API...")
    
    # Scrape the data
    df = scrape_20th_century_timeline()
    print(df.head())
    df.to_csv("datasets/scraped/timeline_raw.csv", index=False)