# main.py
from crawler import crawl
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import store_embeddings, get_collection_stats
from rag import ask


def ingest(url: str, max_depth: int = 1, max_chars: int = 512):
    print(f"\n🌐 Crawling: {url} (depth={max_depth})")
    pages = crawl(url, max_depth=max_depth)
    print(f"✅ Crawled {len(pages)} pages")

    print(f"\nChunking...")
    chunks = chunk_pages(pages, max_chars=max_chars)
    print(f"Created {len(chunks)} chunks")

    print(f"\n🔢 Embedding...")
    embedded = embed_chunks(chunks)
    print(f"Embedded {len(embedded)} chunks")

    print(f"\nStoring in Pinecone...")
    store_embeddings(embedded)
    get_collection_stats()

    print(f"\nIngestion complete!")


def query_mode():
    print("\n💬 Query mode — type 'quit' to exit\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "quit":
            break
        ask(question)


def main():
    print("RAG Pipeline")
    print("─" * 40)
    print("1. Ingest a website")
    print("2. Ask a question")
    print("─" * 40)

    choice = input("Choose (1 or 2): ").strip()

    if choice == "1":
        url = input("Enter URL to crawl: ").strip()
        depth = int(input("Max depth (1 or 2 recommended): ").strip())
        ingest(url, max_depth=depth)

        another = input("\nWant to ask questions now? (y/n): ").strip().lower()
        if another == "y":
            query_mode()

    elif choice == "2":
        query_mode()

    else:
        print("Invalid choice. Please run again and enter 1 or 2.")


if __name__ == "__main__":
    main()
