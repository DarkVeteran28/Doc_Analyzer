def build_sources(retrieved_chunks):
    sources = []
    seen_pages = set()

    for chunk in retrieved_chunks:
        page = chunk["page"]

        if page in seen_pages:
            continue

        seen_pages.add(page)

        sources.append({
            "page": page,
            "text": chunk["text"]
        })

    return sources