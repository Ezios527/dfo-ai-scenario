# -*- coding: utf-8 -*-
"""
scenario_generator.py
======================
Генерирует сценарий развития Дальнего Востока на основе данных из ChromaDB
(коллекция dfo_economy) и GigaChat API.

Использование как модуля:
    from scenario_generator import generate_scenario
    result = generate_scenario("биотех", горизонт_лет=10)
    print(result["сценарий"])
    print(result["источники"])

Использование как самостоятельного скрипта:
    python scenario_generator.py биотех 10

Переменные окружения (читаются из .env через python-dotenv):
    GIGACHAT_CREDENTIALS — Authorization key (Base64) из личного кабинета
                            developers.sber.ru, обязателен.
    Требуется установленный сертификат Минцифры
    (russian_trusted_root_ca_pem.crt) в корне проекта.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import uuid
import requests

from rag_retriever import retrieve

# ---------------------------------------------------------------------------
# КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------
GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"  # для физлиц
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://api.giga.chat/v1/chat/completions"
GIGACHAT_MODEL = "GigaChat-2"  # или "GigaChat-2-Max" для лучшего качества
GIGACHAT_CA_BUNDLE = "russian_trusted_root_ca_pem.crt"  # сертификат Минцифры

# Глобальная переменная для кеширования токена (живёт 30 минут)
_gigachat_token = None

TOP_K_CHUNKS = 15
MIN_HORIZON, MAX_HORIZON = 5, 15
BASE_YEAR = 2025

# Ключевые слова для поиска в ChromaDB по каждому приоритету.
# Формируют поисковый запрос — чем точнее термины, тем релевантнее top-15 чанков.
PRIORITY_QUERIES = {
    "авиация": "авиастроение SSJ-New Суперджет самолёты вертолёты Прогресс Иркут производство",
    "биотех": "биотехнологии фармацевтика биомедицина ИНТЦ Русский научный центр",
    "цифра": "цифровизация искусственный интеллект ИИ цифровые платформы космический мониторинг",
    "туризм": "туризм инвестиции турпоездки гостиницы горнолыжные курорты",
    "судостроение": "судостроение верфь Звезда заказы суда портфель заказов",
    "энергетика": "энергетика газификация ТЭЦ электроэнергия генерация",
}

VALID_PRIORITIES = set(PRIORITY_QUERIES.keys())


def _validate_inputs(приоритет: str, горизонт_лет: int):
    """Проверяет корректность входных аргументов."""
    if приоритет not in VALID_PRIORITIES:
        raise ValueError(
            f"Неизвестный приоритет «{приоритет}». "
            f"Допустимые значения: {sorted(VALID_PRIORITIES)}"
        )
    if not (MIN_HORIZON <= горизонт_лет <= MAX_HORIZON):
        raise ValueError(
            f"горизонт_лет должен быть от {MIN_HORIZON} до {MAX_HORIZON}, "
            f"получено {горизонт_лет}"
        )


def _build_prompt(приоритет: str, горизонт_лет: int, chunks: list[dict]) -> str:
    """Собирает текст промпта для GigaChat из шаблона и найденных чанков."""
    целевой_год = BASE_YEAR + горизонт_лет

    if chunks:
        данные_текст = "\n".join(f"- {c['текст']}" for c in chunks)
    else:
        данные_текст = "(релевантных данных в базе не найдено)"

    prompt = (
        f"Ты — аналитик, пишущий сценарий развития Дальнего Востока до {целевой_год} года. "
        f"Приоритет: {приоритет}.\n"
        f"Вот данные из официальных источников:\n{данные_текст}\n\n"
        f"Напиши сценарий из 3 частей:\n"
        f"1. Текущая ситуация (на основе данных)\n"
        f"2. Что произойдёт, если приоритет будет реализован\n"
        f"3. Риски и что может пойти не так\n\n"
        f"ЖЁСТКИЕ ТРЕБОВАНИЯ:\n"
        f"- Используй ТОЛЬКО факты из предоставленных данных. Не добавляй ничего от себя.\n"
        f"- После КАЖДОГО факта указывай источник в квадратных скобках, "
        f"взятый из поля «Источник:» соответствующего факта. "
        f"Пример: «Запущено производство „Димолегина“ [GxP News]».\n"
        f"- В разделе «Риски» указывай только те риски, которые прямо следуют из данных "
        f"(незавершённые переговоры, зависимость от одного региона, узкая специализация, "
        f"недостаток кадров). НЕ пиши про санкции, пандемию, мировую экономику "
        f"и конкуренцию с другими регионами.\n"
        f"- Если цифра относится ко ВСЕМУ ДФО или ко всем отраслям, а не только "
        f"к выбранному приоритету, обязательно укажи это. "
        f"Пример: «потребность в кадрах под все инвестпроекты ДФО — 100 тыс. человек».\n"
        f"- Если данных по какому-то аспекту нет — пиши «нет данных», не выдумывай.\n"
        f"- Пиши конкретно: называй регионы, проекты, компании, цифры из контекста."
    )
    return prompt


def _get_gigachat_token() -> str | None:
    """
    Получает access token через OAuth GigaChat. Токен живёт 30 минут,
    кешируется в глобальной переменной _gigachat_token.
    """
    global _gigachat_token
    if _gigachat_token:
        return _gigachat_token

    if not GIGACHAT_CREDENTIALS:
        print("[scenario_generator] GIGACHAT_CREDENTIALS не задан в .env")
        return None

    headers = {
        "Authorization": f"Bearer {GIGACHAT_CREDENTIALS}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"scope": GIGACHAT_SCOPE}

    try:
        resp = requests.post(
            GIGACHAT_OAUTH_URL,
            headers=headers,
            data=data,
            timeout=30,
            verify=GIGACHAT_CA_BUNDLE,
        )
        resp.raise_for_status()
        _gigachat_token = resp.json()["access_token"]
        return _gigachat_token
    except Exception as e:
        print(f"[scenario_generator] Ошибка авторизации GigaChat: {e}")
        return None


def _call_gigachat(prompt: str) -> str | None:
    """
    Отправляет prompt в GigaChat и возвращает текст ответа.
    Возвращает None при любой ошибке — вызывающий код решает, что делать.
    """
    token = _get_gigachat_token()
    if not token:
        print("[scenario_generator] GigaChat недоступен (нет токена).")
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GIGACHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты — аналитик регионального развития. Пиши по-русски, фактологично.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }

    try:
        resp = requests.post(
            GIGACHAT_API_URL,
            headers=headers,
            json=body,
            timeout=60,
            verify=GIGACHAT_CA_BUNDLE,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[scenario_generator] Ошибка запроса к GigaChat: {e}")
        return None


def generate_scenario(приоритет: str, горизонт_лет: int = 10) -> dict:
    """
    Основная функция: собирает контекст из ChromaDB и генерирует сценарий
    через GigaChat.

    Возвращает:
        {
            "сценарий": str,          # текст сценария от GigaChat
            "источники": list[str],   # уникальные ссылки на первоисточники
            "приоритет": str,
            "целевой_год": int,
            "число_использованных_чанков": int,
        }
    """
    _validate_inputs(приоритет, горизонт_лет)

    query = PRIORITY_QUERIES[приоритет]
    chunks = retrieve(query, top_k=TOP_K_CHUNKS)

    prompt = _build_prompt(приоритет, горизонт_лет, chunks)

    текст = _call_gigachat(prompt)
    if текст is None:
        текст = (
            "[GigaChat недоступен. Генерация отменена. "
            "Проверьте GIGACHAT_CREDENTIALS в .env и наличие сертификата Минцифры.]"
        )

    источники = sorted({c["источник"] for c in chunks if c["источник"]})

    return {
        "сценарий": текст,
        "источники": источники,
        "приоритет": приоритет,
        "целевой_год": BASE_YEAR + горизонт_лет,
        "число_использованных_чанков": len(chunks),
    }


if __name__ == "__main__":
    приоритет_arg = sys.argv[1] if len(sys.argv) > 1 else "биотех"
    горизонт_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    result = generate_scenario(приоритет_arg, горизонт_arg)

    print("=" * 70)
    print(f"СЦЕНАРИЙ: приоритет «{result['приоритет']}», "
          f"горизонт до {result['целевой_год']} года")
    print(f"(использовано чанков контекста: {result['число_использованных_чанков']})")
    print("=" * 70)
    print(result["сценарий"])
    print()
    print("-" * 70)
    print("ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ:")
    for src in result["источники"]:
        print(f"  - {src}")