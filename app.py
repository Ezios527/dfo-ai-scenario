# -*- coding: utf-8 -*-
"""
app.py
======
Веб-интерфейс проекта «ИИ-сценарист будущего Дальнего Востока»
(краевая конференция «Единый Дальний: от истории освоения к технологическому лидерству»).

Запуск:
    streamlit run app.py

Ожидает, что рядом лежат уже готовые модули проекта:
    rag_retriever.py       — функция retrieve()
    scenario_generator.py  — функция generate_scenario(), словарь PRIORITY_QUERIES
и что база ChromaDB уже загружена скриптом chroma_loader.py (папка ./dfo_chroma_db).
"""

import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Свои модули (должны лежать в той же папке)
from rag_retriever import retrieve
from scenario_generator import generate_scenario, PRIORITY_QUERIES

# ---------------------------------------------------------------------------
# НАСТРОЙКИ СТРАНИЦЫ
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ИИ-сценарист ДФО", layout="wide")

# Подключаем Font Awesome через CDN — иконки вместо эмодзи по всему интерфейсу
st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">',
    unsafe_allow_html=True,
)

DB_PATH = "./dfo_chroma_db"
COLLECTION_NAME = "dfo_economy"
TOTAL_CHUNKS_EXPECTED = 407
BASE_YEAR = 2025

# ---------------------------------------------------------------------------
# КОНСТАНТЫ: ПРИОРИТЕТЫ, ЦВЕТА, ИКОНКИ, КООРДИНАТЫ РЕГИОНОВ
# ---------------------------------------------------------------------------

# Отображаемая метка -> внутренний ключ, который понимает generate_scenario()
PRIORITY_LABELS = {
    "Авиация": "авиация",
    "Биотехнологии": "биотех",
    "Цифровизация": "цифра",
    "Туризм": "туризм",
    "Судостроение": "судостроение",
    "Энергетика": "энергетика",
}

# Иконка Font Awesome для каждого приоритета
PRIORITY_ICONS = {
    "авиация": "fa-plane",
    "биотех": "fa-microscope",
    "цифра": "fa-microchip",
    "туризм": "fa-mountain",
    "судостроение": "fa-ship",
    "энергетика": "fa-bolt",
}

# Цвет точек на карте для каждого приоритета
PRIORITY_COLORS = {
    "биотех": "#2ca02c",        # зелёный
    "авиация": "#1f77b4",       # синий
    "цифра": "#9467bd",         # фиолетовый
    "судостроение": "#7f7f7f",  # серый
    "туризм": "#ff7f0e",        # оранжевый
    "энергетика": "#d62728",    # красный
}

# Координаты центров 11 субъектов ДФО (широта, долгота)
REGION_COORDS = {
    "Приморский край": (45.0, 135.0),
    "Хабаровский край": (54.0, 136.0),
    "Камчатский край": (56.0, 159.0),
    "Амурская область": (53.0, 128.0),
    "Магаданская область": (62.0, 152.0),
    "Сахалинская область": (50.5, 143.0),
    "Еврейская автономная область": (48.5, 132.5),
    "Чукотский автономный округ": (66.0, 173.0),
    "Республика Саха (Якутия)": (66.0, 129.0),
    "Республика Бурятия": (53.0, 108.0),
    "Забайкальский край": (52.0, 116.0),
}

MIN_CHUNKS_WARNING_THRESHOLD = 5  # если чанков меньше — предупреждаем о неполноте


# ---------------------------------------------------------------------------
# ФУНКЦИИ-ПОМОЩНИКИ
# ---------------------------------------------------------------------------

@st.cache_resource
def load_chroma_collection():
    """
    Проверяет, что база ChromaDB существует и коллекция доступна.
    Кешируется на весь сеанс работы сервера (@st.cache_resource),
    чтобы не переоткрывать соединение с базой при каждом действии пользователя.
    Возвращает объект коллекции или None, если база недоступна/пуста.
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        return collection
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_scenario_cached(priority_key: str, horizon: int) -> dict:
    """
    Кешированная обёртка над generate_scenario().
    ttl=3600 (1 час) — чтобы повторные нажатия с теми же параметрами
    не тратили токены GigaChat повторно.
    """
    return generate_scenario(priority_key, горизонт_лет=horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def get_region_counts(priority_key: str, top_k: int = 60) -> dict:
    """
    Возвращает {регион_норм: число релевантных чанков} для выбранного приоритета —
    используется для размера точек на карте и для метрики «Регионов покрыто».
    Использует тот же поисковый запрос, что и generate_scenario (PRIORITY_QUERIES),
    чтобы карта и сценарий были согласованы между собой.
    """
    query = PRIORITY_QUERIES.get(priority_key, priority_key)
    try:
        results = retrieve(query, top_k=top_k, filter_dict={"уровень_агрегации": "регион_ДФО"})
    except Exception:
        return {}

    counts: dict = {}
    for r in results:
        region = r.get("регион")
        if region:
            counts[region] = counts.get(region, 0) + 1
    return counts


def split_scenario_sections(text: str) -> dict:
    """
    Пытается разбить свободный текст LLM на три части по характерным заголовкам
    («текущая ситуация», «что произойдёт», «риски»). Если разбить не получилось —
    возвращает весь текст одним блоком, чтобы интерфейс не падал на нестандартном
    ответе модели.
    """
    section_patterns = [
        (r"текущ[а-я]*\s+ситуац[а-я]*", "Текущая ситуация"),
        (r"что\s+произойд[её]т[а-я]*", "Что произойдёт при реализации приоритета"),
        (r"риск[а-я]*", "Риски"),
    ]

    found = []
    for pattern, label in section_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            # расширяем совпадение до границ всей строки, где оно находится —
            # так в «отрезаемую» часть попадает и нумерация («2.»), и хвост
            # заголовка после ключевых слов («...при реализации приоритета»)
            line_start = text.rfind("\n", 0, match.start()) + 1  # +1: если rfind вернул -1, получим 0
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            found.append((line_start, line_end, label))
    found.sort(key=lambda x: x[0])

    if len(found) < 2:
        return {"Сценарий": text.strip()}

    sections = {}
    for i, (line_start, line_end, label) in enumerate(found):
        content_start = line_end  # тело секции — всё, что идёт ПОСЛЕ строки с заголовком
        content_end = found[i + 1][0] if i + 1 < len(found) else len(text)
        body = text[content_start:content_end].strip()
        sections[label] = body
    return sections


def render_map(priority_key: str, region_counts: dict):
    """
    Строит схематическую карту-план ДФО через plotly.express.scatter —
    самый базовый и универсальный тип графика в Plotly, без каких-либо
    внешних зависимостей.

    Почему не scatter_mapbox / scatter_map / scatter_geo: все три требуют
    подгрузки внешних ресурсов в браузере (тайлы карты, атлас границ стран
    с cdn.plot.ly) и/или зависят от версии JS-бандла Plotly, встроенного во
    фронтенд Streamlit. Любой сбой сети или несовпадение версий там даёт
    либо пустой график, либо (что хуже) полный крах рендера страницы —
    именно это вызывало пустые вкладки. Обычный scatter не тянет из сети
    ничего и работает одинаково в любой версии plotly/streamlit.

    Ось X — долгота, ось Y — широта: расположение точек друг относительно
    друга соответствует реальной географии, просто без подложки-карты снизу.
    """
    rows = []
    for region, (lat, lon) in REGION_COORDS.items():
        rows.append({
            "регион": region,
            "lat": lat,
            "lon": lon,
            "чанков": region_counts.get(region, 0),
        })
    df_map = pd.DataFrame(rows)

    # Точки с нулевым числом чанков всё равно показываем (минимальный размер),
    # чтобы было видно полное покрытие ДФО, а не только «где есть данные».
    df_map["размер_точки"] = df_map["чанков"].apply(lambda x: max(x, 1))

    color = PRIORITY_COLORS.get(priority_key, "#1f77b4")

    fig = px.scatter(
        df_map,
        x="lon",
        y="lat",
        size="размер_точки",
        hover_name="регион",
        hover_data={"lon": False, "lat": False, "размер_точки": False, "чанков": True},
        color_discrete_sequence=[color],
        text="регион",
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(
        height=560,
        margin={"r": 10, "t": 10, "l": 10, "b": 10},
        xaxis_title="Долгота",
        yaxis_title="Широта",
        plot_bgcolor="#f7f7f7",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def format_scenario_txt(result: dict) -> str:
    """Формирует текстовое содержимое для кнопки скачивания (.txt)."""
    lines = [
        f"Сценарий развития Дальнего Востока до {result['целевой_год']} года",
        f"Приоритет: {result['приоритет']}",
        f"Использовано чанков контекста: {result['число_использованных_чанков']}",
        "=" * 60,
        "",
        result["сценарий"],
        "",
        "=" * 60,
        "Использованные источники:",
    ]
    lines += [f"- {src}" for src in result["источники"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# БОКОВАЯ ПАНЕЛЬ
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<h3><i class="fas fa-info-circle"></i> О проекте</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Проект подготовлен для краевой конференции «Единый Дальний: "
        "от истории освоения к технологическому лидерству». "
        "Система строит сценарии развития Дальнего Востока на основе "
        "открытых статистических данных и генеративной модели."
    )

    st.markdown(
        '<i class="fas fa-map-marked-alt"></i> '
        '<a href="https://hkotso.ru/" target="_blank">Сайт колледжа</a>',
        unsafe_allow_html=True,
    )

    # Дата последнего обновления базы — берём время изменения файла базы, если он есть
    if os.path.exists(DB_PATH):
        mtime = os.path.getmtime(DB_PATH)
        last_update = datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M")
    else:
        last_update = "база не найдена"
    st.markdown(
        f'<i class="fas fa-calendar-alt"></i> Обновление базы: {last_update}',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<i class="fas fa-sync-alt"></i> очистка кешированных запросов и сценариев',
        unsafe_allow_html=True,
    )
    if st.button("Очистить кеш"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Кеш очищен. Данные будут пересчитаны при следующем запросе.")


# ---------------------------------------------------------------------------
# ЗАГОЛОВОК
# ---------------------------------------------------------------------------
st.markdown(
    '<h1><i class="fas fa-globe"></i> ИИ-сценарист будущего Дальнего Востока</h1>',
    unsafe_allow_html=True,
)
st.markdown("**Сценарии развития макрорегиона на основе открытых данных**")
st.markdown(
    "Система объединяет поиск по базе из 407 фрагментов официальной статистики "
    "(RAG — retrieval-augmented generation) с генеративной языковой моделью: "
    "сначала находятся релевантные данные по выбранному приоритету, "
    "затем на их основе модель формирует связный текст сценария."
)

# ---------------------------------------------------------------------------
# ПРОВЕРКА, ЧТО БАЗА ЗАГРУЖЕНА
# ---------------------------------------------------------------------------
collection = load_chroma_collection()
if collection is None:
    st.error("База не загружена. Запустите chroma_loader.py")
    st.stop()

collection_count = collection.count()
if collection_count == 0:
    st.error("База не загружена. Запустите chroma_loader.py")
    st.stop()

st.markdown("---")

# ---------------------------------------------------------------------------
# ПАНЕЛЬ ПАРАМЕТРОВ И ЗАПУСК ГЕНЕРАЦИИ
# ---------------------------------------------------------------------------
col_params, col_output = st.columns([1, 2])

with col_params:
    st.markdown(
        '<i class="fas fa-chart-line"></i> <b>Приоритет развития</b>',
        unsafe_allow_html=True,
    )
    priority_display = st.selectbox(
        "Приоритет развития",
        options=list(PRIORITY_LABELS.keys()),
        label_visibility="collapsed",
    )
    priority_key = PRIORITY_LABELS[priority_display]

    st.markdown(
        '<i class="fas fa-calendar-alt"></i> <b>Горизонт планирования (лет)</b>',
        unsafe_allow_html=True,
    )
    horizon = st.slider(
        "Горизонт планирования (лет)",
        min_value=5, max_value=15, value=10, step=1,
        label_visibility="collapsed",
    )
    target_year = BASE_YEAR + horizon
    st.caption(f"Целевой год сценария: {target_year}")

    st.markdown(
        '<i class="fas fa-sync-alt"></i> запуск генерации по выбранным параметрам',
        unsafe_allow_html=True,
    )
    generate_clicked = st.button(
        "Сгенерировать сценарий", type="primary", width="stretch",
    )
if generate_clicked:
    if not os.environ.get("GIGACHAT_CREDENTIALS"):
        st.error("LLM недоступна. Проверьте GIGACHAT_CREDENTIALS.")
    else:
        with st.spinner("Анализирую 407 источников..."):
            try:
                result = get_scenario_cached(priority_key, horizon)
                st.session_state["last_result"] = result
                st.session_state["last_priority_key"] = priority_key
            except Exception as e:
                st.error("LLM недоступна. Проверьте GIGACHAT_CREDENTIALS.")
                st.session_state["last_error"] = str(e)

# Достаём последний результат из состояния сессии (переживает перерисовку страницы)
result = st.session_state.get("last_result")
result_priority_key = st.session_state.get("last_priority_key", priority_key)

# Предупреждение о недостатке данных
if result is not None and result["число_использованных_чанков"] < MIN_CHUNKS_WARNING_THRESHOLD:
    st.warning(
        "По этому приоритету в базе мало данных, сценарий может быть неполным."
    )

# ---------------------------------------------------------------------------
# МЕТРИКИ (правая колонка — «вывод» рядом с панелью параметров)
# ---------------------------------------------------------------------------
region_counts = get_region_counts(result_priority_key) if result else {}
regions_covered = sum(1 for v in region_counts.values() if v > 0)

with col_output:
    st.markdown(
        '<i class="fas fa-info-circle"></i> <b>Метрики</b>', unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m1.metric("Всего чанков в базе", TOTAL_CHUNKS_EXPECTED)
    m2.metric("Использовано в сценарии", result["число_использованных_чанков"] if result else "—")
    m3.metric("Регионов покрыто", regions_covered if result else "—")
    m4.metric("Приоритет", priority_display)

    if result is None:
        st.caption(
            "После генерации здесь появятся метрики конкретного сценария. "
            "Полный текст, карта и источники — во вкладках ниже."
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# ВКЛАДКИ
# ---------------------------------------------------------------------------
tab_scenario, tab_map, tab_sources, tab_about = st.tabs(
    ["Сценарий", "Карта", "Источники", "О системе"]
)

# --- ВКЛАДКА «СЦЕНАРИЙ» -----------------------------------------------------
with tab_scenario:
    st.markdown(
        '<h3><i class="fas fa-file-alt"></i> Сценарий</h3>', unsafe_allow_html=True,
    )
    if result is None:
        st.info("Выберите приоритет и горизонт слева, затем нажмите «Сгенерировать сценарий».")
    else:
        sections = split_scenario_sections(result["сценарий"])
        for label, content in sections.items():
            st.markdown(f"#### {label}")
            st.markdown(content)
            st.markdown("")

        st.download_button(
            label="Скачать сценарий (.txt)",
            data=format_scenario_txt(result),
            file_name=f"scenario_{result_priority_key}_{result['целевой_год']}.txt",
            mime="text/plain",
        )

# --- ВКЛАДКА «КАРТА» ---------------------------------------------------------
with tab_map:
    st.markdown(
        '<h3><i class="fas fa-map-marked-alt"></i> Регионы, для которых есть данные</h3>',
        unsafe_allow_html=True,
    )
    map_priority_key = result_priority_key if result else priority_key
    map_counts = region_counts if result else get_region_counts(priority_key)
    fig_map = render_map(map_priority_key, map_counts)
    st.plotly_chart(fig_map, width="stretch")
    st.caption(
        "Схематическое расположение регионов ДФО по долготе и широте (без "
        "географической подложки). Размер точки — количество релевантных "
        "фрагментов базы по выбранному приоритету. Цвет соответствует приоритету."
    )

# --- ВКЛАДКА «ИСТОЧНИКИ» -----------------------------------------------------
with tab_sources:
    sources = result["источники"] if result else []
    st.markdown(
        f'<h3><i class="fas fa-book"></i> Использованные источники ({len(sources)})</h3>',
        unsafe_allow_html=True,
    )
    if not sources:
        st.info("Источники появятся здесь после генерации сценария.")
    else:
        for url in sources:
            st.markdown(f"[{url}]({url})")

# --- ВКЛАДКА «О СИСТЕМЕ» ------------------------------------------------------
with tab_about:
    st.markdown(
        '<h3><i class="fas fa-info-circle"></i> О системе</h3>', unsafe_allow_html=True,
    )
    st.markdown(
        """
**Архитектура**

Система построена по схеме RAG (retrieval-augmented generation):

1. **База знаний** — 407 текстовых фрагментов (чанков), собранных из открытых
   источников: Минвостокразвития, КРДВ, Росстат (регионы ДФО), итоги ВЭФ,
   отчёты о резидентах ТОР и СПВ. Данные охватывают 2019–2025 годы.
2. **Векторное хранилище** — ChromaDB, эмбеддинги строятся моделью
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. **Поиск (retrieval)** — по текстовому запросу находятся наиболее близкие
   по смыслу фрагменты базы, с возможностью фильтрации по региону, отрасли
   и уровню агрегации.
4. **Генерация** — найденные фрагменты передаются в языковую модель
   GigaChat-2 вместе с инструкцией написать сценарий из трёх частей
   (текущая ситуация, эффект реализации приоритета, риски).

**Ограничения**

- Покрытие данных по приоритетам неравномерно: по авиастроению, судостроению
  и биотехнологиям данных существенно больше, чем по энергетике — там,
  где чанков меньше 5, интерфейс явно предупреждает о неполноте сценария.
- Модель не имеет доступа к данным свежее сентября 2026 года (момент сборки базы).
- Система не заменяет экспертную оценку — сценарии предназначены как
  черновая основа для обсуждения, а не как готовый прогноз.

**Авторы**

Проект подготовлен студентом в рамках краевой конференции
«Единый Дальний: от истории освоения к технологическому лидерству».
        """
    )
