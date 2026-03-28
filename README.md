# Semantic Map of Regional Media Agendas for Central-Bank Communication

Этот репозиторий содержит MVP-пайплайн для построения семантической карты региональной медиаповестки по локальным онлайн-новостям Великобритании. Проект решает прикладную задачу раннего выявления тем, которые могут быть значимы для коммуникации центрального банка на региональном уровне.

Пайплайн работает в пять шагов:

1. очистка корпуса и нормализация временной и региональной привязки;
2. построение дешёвого набора candidate-статей;
3. построение или переиспользование story-level embeddings;
4. оценка экономической релевантности, присвоение тем Bank of England и расчёт регионального индекса;
5. построение итоговых таблиц, карт и графиков.

Текущие публичные результаты относятся к режиму sampled semantic pilot. Это означает, что семантический этап выполнялся на выборке story-level наблюдений, а не на полном корпусе.

## Что находится в репозитории

- `config/` — конфиги пайплайна и профили тем BoE.
- `scripts/` — канонические скрипты пайплайна и сборки презентации.
- `notebooks/` — основной notebook для воспроизведения и просмотра результатов.
- `reports/figures/` — итоговые карты и графики в licence-safe форме.
- `presentation/` — шаблон и финальная публичная версия презентации.
- `data/demo/` — небольшой синтетический демонстрационный набор, созданный автором проекта.
- `data/output/hot_regions_topics.csv` — агрегированная таблица сильнейших регионально-тематических сочетаний без текстовых фрагментов корпуса.
- `legacy/` — устаревшие скрипты и промежуточные материалы, сохранённые как история работы, но не входящие в основной публичный pipeline.

## Что не публикуется в GitHub

В репозитории сознательно не размещаются:

- сырой корпус UKTwitNewsCor;
- производные файлы с полными текстами новостей;
- review-таблицы и snippet-таблицы с прямыми текстовыми фрагментами корпуса;
- полные parquet-артефакты и полные embeddings;
- архивы с данными и другие крупные локальные файлы.

Эти ограничения связаны как с размером артефактов, так и с лицензией исходного корпуса. Подробности приведены в [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## Структура данных и методология

Источник данных — `articles.csv` из открытого корпуса UKTwitNewsCor: более 2,5 млн статей локальных онлайн-медиа Великобритании за январь 2020 — декабрь 2022, собранных по 360 локальным доменам и дополненных региональными и social-media метаданными. Корпус охватывает 94% Local Authority Districts и позволяет анализировать локальные медиаповестки по `LAD / main_LAD`.

Методологически проект измеряет не точную географию событий, а структуру локального медиаполя по административным районам.

Ключевые решения pipeline:

- экономическая релевантность определяется отдельным semantic gate и не приравнивается к одной из 7 тем BoE;
- семантика считается на уровне `story_key`, а затем переносится обратно на статьи;
- синдицированные истории downweight-ятся через `effective_weight = dup_weight / sqrt(region_spread)`;
- региональный индекс строится на сочетании трёх компонент: `surprise`, `momentum` и `mean_similarity`.

## Быстрый старт

Ниже приведён минимальный сценарий для локальной работы из корня будущего репозитория `final_project`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Сценарий 1. Работа с готовыми локальными артефактами

Этот режим подходит, если вне GitHub-репозитория у вас уже существуют локальные артефакты:

- `data/interim/story_lookup.parquet`
- `data/interim/story_embeddings.npy`
- `data/interim/scored_articles.parquet`
- `data/interim/regional_topic_index.parquet`

В этом случае можно:

```powershell
.\.venv\Scripts\python.exe scripts/04_score_and_build_index.py --config config/settings.yaml --model-size small --reuse-embeddings
.\.venv\Scripts\python.exe scripts/05_make_outputs.py --config config/settings.yaml
.\.venv\Scripts\python.exe scripts/build_presentation.py
```

Альтернативно можно открыть `notebooks/mvp_pipeline.ipynb` и выбрать режим `Работа с готовыми локальными артефактами`.

### Сценарий 2. Полный локальный прогон

Полный прогон возможен только после отдельного получения корпуса UKTwitNewsCor в соответствии с его лицензией.

```powershell
.\.venv\Scripts\python.exe scripts/01_clean_articles.py --config config/settings.yaml
.\.venv\Scripts\python.exe scripts/02_filter_candidates.py --config config/settings.yaml
.\.venv\Scripts\python.exe scripts/03_build_embeddings.py --config config/settings.yaml --mode build --model-size small --run-mode dev
.\.venv\Scripts\python.exe scripts/04_score_and_build_index.py --config config/settings.yaml --model-size small
.\.venv\Scripts\python.exe scripts/05_make_outputs.py --config config/settings.yaml
.\.venv\Scripts\python.exe scripts/build_presentation.py
```

## Демонстрационный набор

В репозитории есть синтетический файл [data/demo/demo_articles.csv](data/demo/demo_articles.csv), созданный автором проекта. Он не относится к UKTwitNewsCor и может использоваться для smoke-test запуска.

Для этого подготовлен отдельный конфиг:

```powershell
.\.venv\Scripts\python.exe scripts/01_clean_articles.py --config config/settings.demo.yaml
.\.venv\Scripts\python.exe scripts/02_filter_candidates.py --config config/settings.demo.yaml
.\.venv\Scripts\python.exe scripts/03_build_embeddings.py --config config/settings.demo.yaml --mode build --model-size small --run-mode full
.\.venv\Scripts\python.exe scripts/04_score_and_build_index.py --config config/settings.demo.yaml --model-size small
.\.venv\Scripts\python.exe scripts/05_make_outputs.py --config config/settings.demo.yaml
```

Demo-набор предназначен только для проверки работоспособности кода и не воспроизводит реальные результаты исследования.

## Ключевые артефакты на выходе

Основные локальные артефакты пайплайна:

- `data/interim/story_lookup.parquet` — минимальная story-level таблица для связи embeddings и историй;
- `data/interim/story_embeddings.npy` — story-level embeddings;
- `data/interim/scored_articles.parquet` — итоговая article-level таблица после semantic scoring;
- `data/interim/regional_topic_index.parquet` — регионально-недельная панель с индексом;
- `data/output/hot_regions_topics.csv` — агрегированная таблица топ-регионов и тем;
- `reports/figures/*.png` — карты и тренды;
- `presentation/Semantic_Map_BoE_public.pptx` — финальная публичная презентация.

## Публичная версия и ограничения

Публичная версия репозитория сохраняет только code-first и licence-safe результаты. Это означает:

- figures и агрегированные CSV доступны внутри репозитория;
- полные промежуточные артефакты считаются локальными и не коммитятся;
- notebook очищен от output-ячеек с прямыми текстовыми фрагментами;
- презентация не содержит прямых цитат из корпуса.

Ограничения текущей версии:

- семантический прогон представлен в режиме sampled semantic pilot;
- ручная gold-label калибровка пока не выполнена;
- качество результата чувствительно к выбору semantic thresholds и параметров сглаживания.

## Используемый стек

- Python
- pandas, numpy, pyarrow
- sentence-transformers
- matplotlib, seaborn
- python-pptx, Pillow
- YAML-конфиги

## Лицензия

Код и авторские материалы репозитория распространяются по лицензии MIT. Третьесторонние данные лицензируются отдельно и не входят в действие MIT-лицензии. См. [LICENSE](LICENSE) и [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

