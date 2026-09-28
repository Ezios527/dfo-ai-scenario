# -*- coding: utf-8 -*-
"""
rag_retriever.py
=================
Функция retrieve() для поиска релевантных чанков в коллекции ChromaDB
«dfo_economy», созданной скриптом chroma_loader.py.

Использование как модуля:
    from rag_retriever import retrieve
    results = retrieve("уровень безработицы в Хабаровском крае", top_k=5)

Использование как самостоятельного скрипта — запускает три тестовых запроса:
    python rag_retriever.py
"""

from __future__ import annotations
import os
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./dfo_chroma_db"
COLLECTION_NAME = "dfo_economy"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_client = None
_collection = None


def _get_collection():
    """
    Ленивая инициализация клиента и коллекции — чтобы модель эмбеддингов
    не загружалась при простом импорте модуля, только при первом вызове retrieve().
    """
    global _client, _collection
    if _collection is not None:
        return _collection

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"База ChromaDB не найдена по пути {DB_PATH}. "
            f"Сначала запустите chroma_loader.py для загрузки данных."
        )

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    _client = chromadb.PersistentClient(path=DB_PATH)
    _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def retrieve(query: str, top_k: int = 5, filter_dict: dict | None = None) -> list[dict]:
    """
    Ищет top_k наиболее релевантных чанков по запросу query.

    Параметры:
        query        — текстовый поисковый запрос на русском языке
        top_k        — сколько результатов вернуть
        filter_dict  — опциональный фильтр по метаданным ChromaDB, например:
                        {"уровень_агрегации": "регион_ДФО"}
                        {"отрасль": "промышленность"}
                       Для нескольких условий сразу используйте синтаксис Chroma "$and":
                        {"$and": [{"уровень_агрегации": "регион_ДФО"},
                                   {"отрасль": "промышленность"}]}

    Возвращает список словарей вида:
        {
            "текст": str,
            "регион": str,
            "год": int | None,
            "отрасль": str,
            "источник": str,   # ссылка на первоисточник
            "score": float,    # чем МЕНЬШЕ, тем релевантнее (это расстояние, не сходство)
        }
    """
    collection = _get_collection()

    query_kwargs = {
        "query_texts": [query],
        "n_results": top_k,
    }
    if filter_dict:
        query_kwargs["where"] = filter_dict

    raw = collection.query(**query_kwargs)

    results = []
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        god = meta.get("год_число")
        results.append({
            "текст": doc,
            "регион": meta.get("регион_норм", ""),
            "год": int(god) if isinstance(god, (int, float)) and god != "" else None,
            "отрасль": meta.get("отрасль", ""),
            "источник": meta.get("ссылка", ""),
            "score": round(float(dist), 4),
        })

    return results


def _print_results(query: str, results: list[dict]):
    print(f"\nЗАПРОС: «{query}»")
    print("-" * 70)
    if not results:
        print("Ничего не найдено.")
        return
    for i, r in enumerate(results, 1):
        print(f"{i}. [score={r['score']}] {r['регион']} | {r['отрасль']} | {r['год']}")
        print(f"   {r['текст']}")
        print(f"   источник: {r['источник']}")


if __name__ == "__main__":
    # три тестовых запроса из ТЗ
    test_queries = [
        "уровень безработицы в Хабаровском крае",
        "проекты в судостроении ДФО",
        "население Дальнего Востока",
    ]
    for q in test_queries:
        res = retrieve(q, top_k=5)
        _print_results(q, res)
