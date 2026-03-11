from crawler import crawl
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import store_embeddings, get_collection_stats

pages = crawl("https://www.samsportfolio.xyz", max_depth=1)
chunks = chunk_pages(pages, max_chars=512)
embedded = embed_chunks(chunks)
store_embeddings(embedded)
get_collection_stats()
