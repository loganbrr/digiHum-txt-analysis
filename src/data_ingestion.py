import os
import requests
from bs4 import BeautifulSoup
import pdfplumber
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple
import time

# / logging #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# / data ingestion # 
class FOMCDataIngester:
    def __init__(self, base_url: str = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"):
        self.base_url = base_url
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def get_meeting_links(self) -> list:
        """Scrape FOMC meeting minutes links from the website."""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the most recent PDF link based on release date
            links = []
            current_date = datetime.now()
            min_date_diff = float('inf')
            most_recent_link = None

            for link in soup.find_all('a'):
                href = link.get('href', '')
                if 'fomcminutes' in href and 'pdf' in href:
                    # Get the parent element containing the release date
                    parent = link.parent
                    if parent:
                        # Look for text containing the release date using string instead of text
                        date_text = parent.find(string=lambda s: s and 'Released' in s)
                        if date_text:
                            try:
                                # Extract date from text like "Released March 17, 2023"
                                date_str = date_text.strip().split('Released ')[1]
                                link_date = datetime.strptime(date_str, '%B %d, %Y')
                                date_diff = abs((current_date - link_date).days)
                                
                                if date_diff < min_date_diff:
                                    min_date_diff = date_diff
                                    most_recent_link = href
                            except (ValueError, IndexError):
                                continue
            
            if most_recent_link:
                links.append(most_recent_link)
            return links
        except Exception as e:
            logger.error(f"Error fetching meeting links: {str(e)}")
            return []

    def download_file(self, url: str) -> Optional[Tuple[str, str]]:
        """Download a file from the given URL and return its content and format."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Determine file format
            file_format = 'pdf' if url.endswith('.pdf') else 'html'
            
            # Generate filename based on date
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"fomc_minutes_{date_str}.{file_format}"
            filepath = self.raw_data_dir / filename
            
            # Save the file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded {filename}")
            return str(filepath), file_format
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {str(e)}")
            return None

    def process_file(self, filepath: str, file_format: str) -> Optional[str]:
        """Process the downloaded file and extract text content."""
        try:
            if file_format == 'pdf':
                with pdfplumber.open(filepath) as pdf:
                    text = '\n'.join(page.extract_text() for page in pdf.pages)
            else:  # HTML
                with open(filepath, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text = soup.get_text()
            
            return text
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
            return None

    def run(self):
        """Main execution method to download and process FOMC minutes."""
        links = self.get_meeting_links()
        for link in links:
            result = self.download_file(link)
            if result:
                filepath, file_format = result
                text = self.process_file(filepath, file_format)
                if text:
                    # Save processed text
                    output_path = self.raw_data_dir / f"{Path(filepath).stem}_processed.txt"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    logger.info(f"Successfully processed and saved {output_path}")
            
            # Be nice to the server
            time.sleep(1)

if __name__ == "__main__":
    ingester = FOMCDataIngester()
    ingester.run()