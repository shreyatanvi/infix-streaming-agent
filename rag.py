from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import re

from langchain_core.documents import Document

KB_PATH = Path(__file__).with_name("knowledge.json")
WORD_RE = re.compile(r"\b\w+\b")
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "what",
    "your",
    "do",
    "i",
    "me",
    "about",
    "tell",
    "and",
    "or",
    "policy",
}


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text) if token.lower() not in STOPWORDS}


@dataclass
class KnowledgeBase:
    documents: list[Document]
    vectorstore: object | None = None

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        if self.vectorstore is not None:
            try:
                return self.vectorstore.similarity_search(query, k=k)
            except Exception:
                pass

        query_terms = _tokenize(query)
        scored: list[tuple[int, Document]] = []
        for document in self.documents:
            content_terms = _tokenize(document.page_content)
            title_terms = _tokenize(str(document.metadata.get("title", "")))
            category_terms = _tokenize(str(document.metadata.get("category", "")))
            score = len(query_terms & content_terms)
            score += 2 * len(query_terms & title_terms)
            score += len(query_terms & category_terms)
            scored.append((score, document))

        ranked = [doc for score, doc in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
        return ranked[:k] or self.documents[:k]


def create_rag() -> KnowledgeBase:
    with KB_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    documents = [
        Document(
            page_content=record["content"],
            metadata={
                "id": record["id"],
                "category": record["category"],
                "title": record["title"],
            },
        )
        for record in records
    ]

    vectorstore = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore = FAISS.from_documents(documents, embeddings)
        except Exception:
            vectorstore = None

    return KnowledgeBase(documents=documents, vectorstore=vectorstore)


def retrieve_context(kb: KnowledgeBase, query: str, k: int = 3) -> tuple[list[dict[str, str]], str]:
    docs = kb.retrieve(query, k=k)
    normalized = [
        {
            "title": str(doc.metadata.get("title", "Knowledge")),
            "category": str(doc.metadata.get("category", "general")),
            "content": doc.page_content,
        }
        for doc in docs
    ]
    context = "\n".join(f"- {item['title']}: {item['content']}" for item in normalized)
    return normalized, context
