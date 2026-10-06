"""
Text extraction and HTML sanitization for news articles.
"""

import re

from bs4 import BeautifulSoup


class TextExtractor:
    def clean_html(self, raw_html: str) -> str:
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


text_extractor = TextExtractor()
