def chunk_pages(pages, chunk_size=600, overlap=100):
    chunks = []

    for page in pages:
        words = page["text"].split()

        start = 0
        chunk_number = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]

            chunk = {
                "chunk_id": f"page_{page['page']}_chunk_{chunk_number}",
                "page": page["page"],
                "text": " ".join(chunk_words)
            }

            chunks.append(chunk)

            chunk_number += 1

            start += chunk_size - overlap

    return chunks