# Контракт вимірювання

Основна ціль — committed output tokens/s одного запиту. Звичайний server step, speculative draft token і SSE chunk не є взаємозамінними одиницями.

Записувати окремо: TTFT (client end-to-end), server prefill, output decode, p50/p95 inter-token latency, загальний wall time, accepted/emitted tokens. Для speculation збирати draft/verify/replay time, acceptance і actual traffic. При формулі decode rate явно фіксувати, чи перший token виключено з numerator/denominator; використовувати однакову формулу у всіх режимах.

Фіксовані checkpoint/tokenizer/templates, seed/sampling, actual token counts, prompts, runtime/driver, expert cache і PLE. Greedy контроль і normal-sampling correctness окремо. EOS-shortened runs не порівнювати як однаковий output budget. Жодних усереднень tokens/s різних довжин без raw samples.

Етапи: cold load/JIT; warmup; 3 paired повтори на 4K/16K/32K; optional 64K після fit. На фінальному preset >=15 min sustained run, temperatures/clocks/power і throttling. Не падати в swap для отримання штучно великого контексту. Не очищати системний page cache глобально; cold runs організувати контрольовано без втручання в сторонні процеси.

Виділити: baseline main, best-known config без code changes, optimized branch. Для KV — fixed expert slots і окремо retuned expert budget. Для CPU/GPU — isolated і contended bandwidth, але висновок за full-model wall time.

Початковий gate code optimization: >=5% median committed throughput gain вище шуму, <=5% p95 latency regression на declared workload, correctness без failures. До tuning узгодити suite/пороги в задачі 03; не змінювати їх post-hoc. Якщо виграш лише на певному context/workload, preset явно обмежений ним. TTFT не маскувати decode speedup.

Quality: greedy parity де semantic-equivalent, exact speculative verification invariants, multi-depth retrieval, JSON/tool-call і fixed coding checks. Чисельні kernel tolerances беруться з обґрунтованих reference tests. NLL/PPL потребує окремого teacher-forced model runner; до його появи — blocked, а не zero degradation. Якщо змінюється checkpoint/weight precision — новий quality baseline, не чистий runtime A/B.

Приблизна діагностика: transfer floor = measured miss bytes per emitted token / measured gather bytes/s. Це нижня оцінка transfer cost, не прогноз загальних tokens/s; routing distribution, overlapping CPU/GPU, PLE і serial layers враховуються trace. Накладені ділянки не підсумовувати двічі.

Report: plan task ID, git SHA, dirty diff hash, model revision, hardware fingerprint без serials, команда, settings, raw timings, memory bytes, TPOT/TTFT, proposed/accepted/emitted tokens, misses/traffic, PLE cache/read metrics, errors/skips і quality status. Профайлер вимкнений у підсумкових speed runs або overhead обґрунтовано.
