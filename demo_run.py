# -*- coding: utf-8 -*-
"""
demo_run.py
===========
Полный тестовый прогон RAG-системы по экономике ДФО:
  1. Загрузка rag_chunks.csv в ChromaDB (chroma_loader.py)
  2. Три тестовых поисковых запроса (rag_retriever.py)
  3. Один тестовый сценарий по приоритету «биотех» на 10 лет (scenario_generator.py)

Запуск:
    python demo_run.py

Перед первым запуском убедитесь, что rag_chunks.csv лежит в этой же папке.
"""

from __future__ import annotations
import subprocess
import sys

from rag_retriever import retrieve, _print_results
from scenario_generator import generate_scenario


def step_1_load():
    print("\n" + "#" * 70)
    print("# ШАГ 1. ЗАГРУЗКА ДАННЫХ В CHROMADB")
    print("#" * 70)
    # запускаем chroma_loader.py как отдельный процесс, чтобы не тянуть
    # его побочные эффекты (print-сводку) в текущий namespace
    subprocess.run([sys.executable, "chroma_loader.py"], check=True)


def step_2_test_queries():
    print("\n" + "#" * 70)
    print("# ШАГ 2. ТЕСТОВЫЕ ПОИСКОВЫЕ ЗАПРОСЫ")
    print("#" * 70)

    queries = [
        "уровень безработицы в Хабаровском крае",
        "проекты в судостроении ДФО",
        "население Дальнего Востока",
    ]
    for q in queries:
        results = retrieve(q, top_k=5)
        _print_results(q, results)


def step_3_test_scenario():
    print("\n" + "#" * 70)
    print("# ШАГ 3. ТЕСТОВЫЙ СЦЕНАРИЙ: приоритет «биотех», горизонт 10 лет")
    print("#" * 70)

    result = generate_scenario("биотех", горизонт_лет=10)

    print(f"\nЦелевой год сценария: {result['целевой_год']}")
    print(f"Использовано чанков контекста: {result['число_использованных_чанков']}\n")
    print(result["сценарий"])
    print("\nИсточники:")
    for src in result["источники"]:
        print(f"  - {src}")


if __name__ == "__main__":
    step_1_load()
    step_2_test_queries()
    step_3_test_scenario()
