from crawler import crawl
from chunker import chunk_pages

pages = crawl("https://www.samsportfolio.xyz", max_depth=1)
chunks = chunk_pages(pages)

print(f"Total chunks: {len(chunks)}")
print(chunks[0])
