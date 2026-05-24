"""Article scraping for the live demo.

Best-effort extraction of the main text body from a public news URL using
BeautifulSoup heuristics. Designed to be small, dependency-light, and to fail
loudly when a site blocks scraping rather than to silently return garbage.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
import re

from bs4 import BeautifulSoup

from nlp_project.config import ScrapeConfig
from nlp_project.preprocess import normalize_article


class ScrapeError(RuntimeError):
    """Raised when an article cannot be fetched or parsed."""


@dataclass(frozen=True)
class ScrapedArticle:
    url: str
    title: str
    article: str
    domain: str


def fetch_article(url: str, scrape_config: ScrapeConfig) -> ScrapedArticle:
    import requests

    try:
        response = requests.get(
            url,
            headers={"User-Agent": scrape_config.user_agent},
            timeout=scrape_config.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ScrapeError(f"Could not fetch {url}: {exc}") from exc

    return parse_article(url, response.text)


def parse_article(url: str, html: str) -> ScrapedArticle:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "aside",
                     "footer", "header", "nav", "form"]):
        tag.decompose()

    title = _extract_title(soup)
    body = _extract_body(soup)
    body = normalize_article(body)
    if len(body.split()) < 30:
        raise ScrapeError(
            "Parsed article appears too short — the page may be paywalled "
            "or require JavaScript rendering."
        )

    domain = urlparse(url).netloc.lower()
    return ScrapedArticle(url=url, title=title, article=body, domain=domain)


def _extract_title(soup: BeautifulSoup) -> str:
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return str(og_title["content"]).strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return ""


def _extract_body(soup: BeautifulSoup) -> str:
    article = soup.find("article")
    candidates: list[str] = []
    if article:
        candidates.extend(p.get_text(" ", strip=True) for p in article.find_all("p"))
    if not candidates:
        main = soup.find("main") or soup.body or soup
        candidates.extend(p.get_text(" ", strip=True) for p in main.find_all("p"))

    cleaned = [c for c in candidates if len(c.split()) >= 8 and not _looks_like_chrome(c)]
    return " ".join(cleaned)


def _looks_like_chrome(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "subscribe to",
            "sign up for",
            "follow us on",
            "share this article",
            "cookie policy",
            "privacy policy",
        )
    ) or bool(re.match(r"^(by\s+|copyright|©)", lowered))
