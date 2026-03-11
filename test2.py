from crawler import crawl
from chunker import chunk_pages
from embedder import embed_chunks

pages = crawl("https://www.samsportfolio.xyz", max_depth=1)
chunks = chunk_pages(pages, max_chars=512)
embedded = embed_chunks(chunks)

print(f"Total embedded chunks: {len(embedded)}")
