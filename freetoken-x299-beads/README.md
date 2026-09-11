# Оновлення v2

20 великих задач розділено на 40: фаза a фіксує контракт/fixtures, фаза b виконує й перевіряє. Нормативні документи: CONTRACTS.md, GATES.md, RUNBOOK.md, CPU_BRANCHES.md. Для вже імпортованого v1 спочатку MIGRATION.md. Новий hardware collector лише збирає дані; він не визначає memory channels за назвою плати.

# FreeToken: Qwen3.8-Flash-Next на RTX3090 / X299

Версія 2. Окремий пакет для Beads: 40 задач, один epic, blocking dependencies, acceptance criteria і промпт агента. Ціль — швидкість генерації одного запиту на RTX3090, PCIe 3.0 x16, 104 GB DDR4-2133, X299 та i7.

Підготовлено 2026-09-10 за fork SHA `b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d`. Це новіша база, ніж у попередньому KV-архіві: у fork уже є KV kernels, benchmark, MTP і prompt lookup. Не реалізовуйте їх заново.

## Імпорт

Розпакуйте `freetoken-x299-beads` у корінь FreeToken. Наявний `.beads` і задачі попереднього пакета збережіть. Потрібні Python3 і встановлений/ініціалізований `bd`.

```bash
bd prime
python3 freetoken-x299-beads/import_beads.py
python3 freetoken-x299-beads/import_beads.py --apply
bd ready
```

Без `--apply` виконується тільки перевірка DAG і preview. Імпорт створює новий epic через `bd create`, а dependencies через `bd dep add`. `plan.json` є portable manifest пакета, а не JSONL базою Beads. Після імпорту джерелом статусів є Beads; task markdown — лише вхідні описи, не паралельний tracker.

Скрипт не змінює AGENTS.md, не робить commit/push і не публікує GitHub issues. `.import-state.json` у цьому каталозі зберігає IDs і прогрес. Не видаляйте його після часткового імпорту; повторний запуск перевіряє вже створені IDs. Якщо є `pending`, звірте останню операцію через `bd list`/`bd show`, доповніть state ID/edge і лише після цього очистьте pending. Не запускайте імпортер паралельно. Не починайте виконання задач до завершення імпорту всіх залежностей.

Передайте агенту `AGENT_PROMPT.md`. Прочитайте `PLAN.uk.md` для архітектурних висновків, `BENCHMARK_CONTRACT.md` для метрик.

## Порядок

| Задачі | Результат |
|---|---|
| 01–04 | Exact hardware/checkpoint, baseline і bottleneck traces |
| 05–08 | Найкращі налаштування, CPU affinity, memory split, валідний hardware profile |
| 09–14 | Вимірювані runtime оптимізації hybrid/cache/PCIe/PLE/kernels |
| 15–17 | Приймання й tuning наявних MTP та prompt lookup |
| 18–19 | Спільний KV/expert tuning та interactivity |
| 20 | Фінальні виміряні presets і regression report |

Для задач з експериментальними гіпотезами допустимий результат no-change з доказами: план не вимагає впроваджувати оптимізацію, яка не прискорює вашу машину. Відсутність hardware measurements — blocked, не no-change.

Перевірено синтаксис імпортера, файли manifest, DAG, точки інтеграції в repo tree і цілісність архіву. Реальний `bd --apply` та hardware benchmarks тут не запускали: немає `bd` і доступу до описаного сервера. Прогноз tokens/s навмисно не видається за результат.
