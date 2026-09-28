# -*- coding: utf-8 -*-
"""
chroma_loader.py
=================
Загружает датасет rag_chunks.csv (407 чанков по экономике ДФО) в локальную
persistent-базу ChromaDB. Скрипт идемпотентен: повторный запуск не создаёт
дублей — используется upsert по id из CSV (столбец id -> id записи в Chroma).

Запуск:
    python chroma_loader.py [путь_к_csv]

По умолчанию путь к CSV — "./rag_chunks.csv", база создаётся в "./dfo_chroma_db".

Зависимости:
    pip install chromadb sentence-transformers pandas --break-system-packages
"""

from __future__ import annotations
import os
import sys
import math

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "./rag_chunks.csv"
DB_PATH = "./dfo_chroma_db"
COLLECTION_NAME = "dfo_economy"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Метаданные, которые сохраняем для каждого чанка (для фильтрации при поиске)
METADATA_COLUMNS = [
    "регион_норм", "уровень_агрегации", "год_число",
    "период", "отрасль", "полнота", "ссылка", "датасет",
]


def clean_metadata_value(value):
    """
    ChromaDB принимает в metadata только str / int / float / bool
    (None и NaN не допускаются). Приводим значения к безопасному виду.
    """
    if value is None:
        return ""
    # pandas NaN (float('nan')) — заменяем на пустую строку
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", "<na>", ""}:
        return ""
    # год_число храним как int, если это возможно
    if isinstance(value, (int, float)):
        try:
            if float(value).is_integer():
                return int(value)
        except (ValueError, OverflowError):
            pass
        return float(value)
    return str(value)


def build_metadata(row: pd.Series) -> dict:
    """Собирает словарь метаданных для одной строки CSV."""
    return {col: clean_metadata_value(row.get(col)) for col in METADATA_COLUMNS}


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Не найден файл {csv_path}. Убедитесь, что rag_chunks.csv лежит рядом "
            f"со скриптом, или укажите путь первым аргументом командной строки."
        )
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required_cols = {"id", "текст_для_rag"} | set(METADATA_COLUMNS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют обязательные колонки: {missing}")
    return df


def get_embedding_function():
    """
    Инициализирует функцию эмбеддингов на базе sentence-transformers.
    При первом запуске модель (~470 МБ) скачивается с huggingface.co —
    для этого нужен доступ в интернет.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def main():
    print(f"Читаю датасет: {CSV_PATH}")
    df = load_dataset(CSV_PATH)
    print(f"Загружено строк из CSV: {len(df)}")

    print(f"Инициализирую модель эмбеддингов: {EMBEDDING_MODEL}")
    print("(при первом запуске модель скачивается из интернета, это может занять пару минут)")
    embed_fn = get_embedding_function()

    print(f"Открываю/создаю persistent-базу ChromaDB: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"description": "Экономика ДФО 2019-2025 — база знаний для RAG-сценариев"},
    )

    ids = df["id"].astype(str).tolist()
    documents = df["текст_для_rag"].astype(str).tolist()
    metadatas = [build_metadata(row) for _, row in df.iterrows()]

    # --- ИДЕМПОТЕНТНАЯ ЗАГРУЗКА -------------------------------------------
    # upsert: если id уже существует — запись обновится, а не задублируется.
    # Если в установленной версии chromadb нет upsert (старые версии),
    # используем ручной фолбэк: сначала удаляем существующие id, затем add().
    print(f"Загружаю {len(ids)} записей в коллекцию «{COLLECTION_NAME}» (upsert)...")

    BATCH_SIZE = 100  # эмбеддинг-модели обычно ограничивают размер батча
    if hasattr(collection, "upsert"):
        for start in range(0, len(ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
    else:
        # Фолбэк для старых версий chromadb без upsert
        existing = set()
        try:
            existing = set(collection.get(ids=ids)["ids"])
        except Exception:
            pass
        if existing:
            collection.delete(ids=list(existing))
        for start in range(0, len(ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    print("Загрузка завершена.\n")

    # ---------------------------------------------------------------------
    # ИТОГОВАЯ СВОДКА
    # ---------------------------------------------------------------------
    total = collection.count()
    print("=" * 60)
    print("СВОДКА ПО КОЛЛЕКЦИИ")
    print("=" * 60)
    print(f"Всего записей в коллекции «{COLLECTION_NAME}»: {total}")

    print("\nРаспределение по датасету (yearly / quarterly / qualitative):")
    print(df["датасет"].value_counts().to_string())

    print("\nРаспределение по регионам (регион_норм):")
    print(df["регион_норм"].value_counts().to_string())

    print(f"\nБаза сохранена в: {os.path.abspath(DB_PATH)}")


if __name__ == "__main__":
    main()
