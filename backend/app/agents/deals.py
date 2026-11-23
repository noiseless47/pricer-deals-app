from pydantic import BaseModel, Field
from typing import List, Dict, Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time
import logging


logger = logging.getLogger(__name__)

feeds = [
    ("DealNews · latest", "https://www.dealnews.com/?rss=1&sort=time"),
    (
        "DealNews · video games",
        "https://www.dealnews.com/c191/Gaming-Toys/Video-Games/?rss=1",
    ),
    (
        "Slickdeals · front page",
        "https://slickdeals.net/newsearch.php?mode=frontpage&rss=1&searcharea=deals&searchin=first",
    ),
    ("TechBargains · latest", "https://www.techbargains.com/rss.xml"),
]


def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = str(entry.get("title") or "").strip()
        self.summary = extract(str(entry.get("summary") or entry.get("description") or ""))
        links = entry.get("links") or []
        self.url = str(
            next((link.get("href") for link in links if link.get("href")), None)
            or entry.get("link")
            or ""
        ).strip()
        if not self.title or not self.url:
            raise ValueError("RSS entry is missing a title or URL")

        response = requests.get(
            self.url,
            timeout=(5, 20),
            headers={"User-Agent": "Mozilla/5.0 (compatible; Pricer/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        content_section = soup.find("div", class_="content-section")
        content = content_section.get_text(" ", strip=True) if content_section else ""
        content = content.replace("\nmore", "").replace("\n", " ")
        if not content:
            # Deal pages can change their markup; use the RSS summary rather
            # than failing the entire scan or sending navigation text to the LLM.
            content = self.summary
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
        self.truncate()

    def truncate(self):
        """
        Limit the fields to a sensible length to avoid sending too much info to the model
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @staticmethod
    def matches_search(entry: Dict[str, str], search_query: str) -> bool:
        """Match every query word against lightweight RSS text fields."""
        terms = re.findall(r"[\w'-]+", search_query.casefold())
        if not terms:
            return True
        title = str(entry.get("title") or "")
        summary = str(entry.get("summary") or entry.get("description") or "")
        searchable_text = BeautifulSoup(f"{title} {summary}", "html.parser").get_text(
            " ", strip=True
        ).casefold()

        normalized_query = " ".join(terms)
        platform_aliases = {
            "playstation 5": ("playstation 5", "ps5"),
            "ps5": ("ps5", "playstation 5"),
            "playstation 4": ("playstation 4", "ps4"),
            "ps4": ("ps4", "playstation 4"),
            "xbox series x": ("xbox series x", "series x"),
            "series x": ("xbox series x", "series x"),
            "xbox series s": ("xbox series s", "series s"),
            "series s": ("xbox series s", "series s"),
        }
        if normalized_query in platform_aliases:
            return any(alias in searchable_text for alias in platform_aliases[normalized_query])
        return all(term in searchable_text for term in terms)

    @classmethod
    def fetch(cls, show_progress: bool = False, search_query: str = "") -> List[Self]:
        """
        Retrieve deals from RSS feeds, filtering by query before opening pages.
        """
        search_query = search_query.strip()
        deals = []
        feed_sources = list(feeds)
        if search_query:
            search_url = requests.Request(
                "GET",
                "https://slickdeals.net/newsearch.php",
                params={
                    "q": search_query,
                    "rss": "1",
                    "searcharea": "deals",
                    "searchin": "first",
                },
            ).prepare().url
            feed_sources.insert(0, ("Slickdeals · keyword search", search_url))

        feed_iter = tqdm(feed_sources) if show_progress else feed_sources
        seen_urls = set()
        for source_name, feed_url in feed_iter:
            try:
                response = requests.get(
                    feed_url,
                    timeout=(5, 20),
                    headers={"User-Agent": "Mozilla/5.0 (compatible; Pricer/1.0)"},
                )
                response.raise_for_status()
                feed = feedparser.parse(response.content)
            except requests.RequestException as exc:
                logger.warning("Could not fetch %s feed: %s", source_name, exc)
                continue
            entries = feed.entries if search_query else feed.entries[:50]
            if search_query:
                entries = [entry for entry in entries if cls.matches_search(entry, search_query)]
            logger.info(
                "%s feed has %s entries matching %r",
                source_name,
                len(entries),
                search_query or "all deals",
            )
            for entry in entries[:8]:
                links = entry.get("links") or []
                entry_url = str(
                    next((link.get("href") for link in links if link.get("href")), None)
                    or entry.get("link")
                    or ""
                ).strip()
                if entry_url and entry_url in seen_urls:
                    continue
                try:
                    deal = cls(entry)
                    deals.append(deal)
                    seen_urls.add(deal.url)
                except Exception as exc:
                    title = entry.get("title", "(untitled)")
                    logger.warning("Skipping deal feed entry %r: %s", title, exc)
                time.sleep(0.05)
        return deals


class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """

    deal: Deal
    estimate: float
    discount: float
