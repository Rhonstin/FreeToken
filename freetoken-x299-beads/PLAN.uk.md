# План оптимізації швидкості генерації

## Рішення

Почати з actual decode traces і best-known конфігурації. На цій машині найбільш перспективні гіпотези — менше expert bytes через PCIe, більша корисна GPU residency, точніший CPU/GPU split під DDR4-2133 та менші PLE stalls. MTP може амортизувати дорогий target pass, але його BF16 draft experts і verify/replay також коштують RAM, PCIe та часу. Пріоритет визначають виміри, не назва технології.

## Відоме і невідоме

| Параметр | Стан |
|---|---|
| GPU | RTX3090; планувати SM86 і перевірити доступну VRAM |
| Link | Заявлено PCIe3 x16; перевірити negotiated width/speed під load |
| RAM | Заявлено 104 GB DDR4-2133; перевірити configured MT/s, layout і available bytes |
| Platform | X299, i7; exact SKU/cores/ISA/channels невідомі |
| Storage | Не задано; не припускати NVMe, SSD або конкретну latency |
| Model | Qwen3.8-Flash-Next; checkpoint/quantization/revision невідомі |
| OS | Не задано; Linux baseline умовний, WSL pin limits перевіряти окремо |

Обсяг 104 GB не визначає bandwidth. Асиметричне заповнення DIMM може мати різні interleave regions; не робити висновок про quad-channel лише з X299. DDR4-2133 не означає 2133 MHz фізичного такту. Налаштування CPU ISA, потоків або DIMM не вгадувати.

## Висновки з актуального коду

1. `moe/benchbw.py` уже вимірює CPU-MoE і PCIe gather разом. `bench_profile.py` використовує contended fraction `pcie_overlap / (pcie_overlap + cpu_overlap)` із fallback для старих профілів. Тому задача — покращити калібрування/validation за CPU, DIMM, PCIe і expert geometry, а не додати overlap benchmark з нуля.
2. `CpuMoeExecutor` уже pin-ить workers і обирає physical cores; має coordination/core reserve поведінку. Треба перевірити її на конкретному i7 і конкуренцію з PLE I/O. `_WFMT_IDS` не містить block-FP8, навіть якщо інші таблиці профілю згадують його: доступність CPU backend перевіряти end-to-end.
3. `OffloadMoeCache` має LRU, hybrid fetch cap, per-layer stats, double-buffer prefill і workaround для small mixed batched copies. Нові policy/transfer оптимізації мають зберегти ці гарантії.
4. `ple_disk.py` уже має pinned staging, token D2H readback і graph flag-sync. C++ reader deduplicates у межах fill; між fill RAM row cache у прочитаній реалізації відсутній. Bounded row cache — кандидат після locality trace, не гарантований speedup.
5. `engine/mtp.py` і `models/qwen4_exp/mtp.py` уже реалізують draft/verify/commit wiring. Для перевіреного у коді RadixArk checkpoint MTP experts BF16 і потребують dedicated banks/cache, відмінних від NVFP4 target. Не узагальнювати це на кожний checkpoint без inspection.
6. `AdaptiveDepthPolicy` реагує на acceptance. Для повільної RAM/PCIe потрібна перевірка cost-aware depth з урахуванням draft і replay часу. Prompt lookup уже існує та взаємовиключний із MTP.
7. `benchmarks/bench_kv_quant.py` — наявний paired harness. Він визнає відсутність NLL/PPL через server API. Розширити його та не підміняти greedy-prefix similarity повноцінною якістю.

## Взаємодія з попереднім KV планом

Цей пакет не дублює створення FP8/NVFP4 codec. Задача 18 перевіряє фактичні стани існуючих Beads і вимірює decode ефект від іншого memory split. Зовнішні Beads IDs не вигадані: агент знаходить їх у вашій БД і за потреби додає справжні dependencies. Наявність коду в tree не означає пройдене RTX3090 acceptance.

## Чого не робити без даних

- Не вмикати hybrid лише тому, що CPU bandwidth більший за PCIe.
- Не pin-ити весь PLE у RAM без peak memory accounting.
- Не зменшувати router top-k, QSA selection або model precision для прихованого speedup.
- Не вмикати native Blackwell/FP8 matmul path на Ampere за назвою weight format.
- Не максимізувати batch/context на шкоду single-user TPOT.
- Не міняти BIOS, power limit або clock settings автоматично. Спочатку тільки діагностика throttling; undervolt/overclock не є задачами цього пакета.

## Джерела та актуальність

Baseline: [b647a420](https://github.com/Rhonstin/FreeToken/commit/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d), перевірено 2026-09-10. Посилання нижче зафіксовані на ньому:

- [Offload cache](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/moe/offload_cache.py)
- [CPU executor](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/moe/cpu_executor.py)
- [Bandwidth profile](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/moe/bench_profile.py)
- [Bandwidth benchmark](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/moe/benchbw.py)
- [PLE disk](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/models/qwen4_exp/ple_disk.py)
- [PLE C++ reader](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp)
- [MTP driver](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/engine/mtp.py)
- [MTP model](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/models/qwen4_exp/mtp.py)
- [CLI args](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/python/freetoken/server/args.py)
- [Existing harness](https://github.com/Rhonstin/FreeToken/blob/b647a420cea3e4c4dc1c6660d8cfe1a2b5bff54d/benchmarks/bench_kv_quant.py)

Пошук upstream виявив відкриті кандидати [#278](https://github.com/FlashML-org/FreeToken/pull/278), [#37](https://github.com/FlashML-org/FreeToken/pull/37), [#38](https://github.com/FlashML-org/FreeToken/pull/38), [#39](https://github.com/FlashML-org/FreeToken/pull/39) для benchmark policy. Це результати пошуку за назвами, не перевірена рекомендація cherry-pick: повний diff/reviews і перетин із fork входять у задачу 08. Жодні авторські speed numbers не прийняті як виміри вашого сервера.
