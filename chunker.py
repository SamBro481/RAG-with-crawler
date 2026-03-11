# chunker.py
import nltk
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


def chunk_text(
    text: str,
    max_chars: int = 512,
    overlap_sentences: int = 1
) -> list[str]:
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_len = len(sentence)

        # If a single sentence exceeds max_chars, include it alone
        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(sentence)
            i += 1
            continue

        # If adding this sentence exceeds the limit, save chunk and overlap
        if current_length + sentence_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep last N sentences as start of next chunk
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len
        i += 1

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_pages(pages: dict[str, str], max_chars: int = 256, overlap_sentences: int = 1) -> list[dict]:
    all_chunks = []

    for url, text in pages.items():
        chunks = chunk_text(text, max_chars, overlap_sentences)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "url": url,
                "chunk_index": i,
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    sample = """
    Artificial intelligence is transforming the world. It powers everything from search engines to self-driving cars.
    Machine learning is a subset of AI. It allows systems to learn from data without being explicitly programmed.
    Deep learning uses neural networks with many layers. These networks can recognize images, speech, and text with high accuracy.
    Natural language processing enables machines to understand human language. It is used in chatbots, translation, and summarization.
    """

    chunks = chunk_text(sample)
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"[Chunk {i}] ({len(chunk)} chars)\n{chunk}\n")
