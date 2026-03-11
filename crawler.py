# crawler.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def scrape_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)

    return clean_text


def get_links(url: str, base_domain: str) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(url, href)         # handle relative URLs e.g. /about
        parsed = urlparse(full_url)

        # Stay on same domain, only http/https, no anchors or file extensions
        if (
            parsed.netloc == base_domain
            and parsed.scheme in ("http", "https")
            and not parsed.fragment                         # skip #section links
            and not full_url.endswith((".pdf", ".jpg", ".png", ".zip"))
        ):
            links.append(full_url.split("#")[0])            # strip any trailing anchors

    return list(set(links))                                 # deduplicate


def crawl(start_url: str, max_depth: int = 2) -> dict[str, str]:
    base_domain = urlparse(start_url).netloc
    visited = set()
    results = {}                                            # { url: text }

    def _crawl(url: str, depth: int):
        if depth > max_depth or url in visited:
            return

        print(f"[depth {depth}] Crawling: {url}")
        visited.add(url)

        try:
            text = scrape_page(url)
            results[url] = text

            if depth < max_depth:
                links = get_links(url, base_domain)
                for link in links:
                    _crawl(link, depth + 1)

        except Exception as e:
            print(f"  ⚠ Skipping {url} — {e}")

    _crawl(start_url, depth=0)
    return results


if __name__ == "__main__":
    url = input("Enter a starting URL: ")
    depth = int(input("Max depth (recommended 1 or 2): "))

    pages = crawl(url, max_depth=depth)

    print(f"\nCrawled {len(pages)} pages")
    for page_url, text in pages.items():
        print(f"\n--- {page_url} ---")
        print(text[:500])
