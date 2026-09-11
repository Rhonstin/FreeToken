# Відтворюваний runbook

Виконувати на target сервері з кореня FreeToken. Не на машині автора плану.

```bash
python3 freetoken-x299-beads/collect_hardware.py --out evidence/x299/hardware.json
bd prime
git rev-parse HEAD
git status --short
uv run ft --version
uv run ft serve --help
uv run ft bench bw --help
uv run python benchmarks/bench_kv_quant.py --help
uv run python benchmarks/bench_decode_moe.py --help
```

Це точні команди для capability discovery. Test path/benchmark options підбираються лише з поточного --help/source; не виконувати placeholders як literals. Задача a записує остаточні argv у commands.json до production change. Збіг pipeline args із старим README не вважати доказом підтримки.

Bandwidth baseline:

```bash
uv run ft bench bw
```

Ця команда вимірює hardware і може записувати hardware profile. Зберегти старий profile перед запуском, зафіксувати новий; не називати її read-only. Один benchmark одночасно. Для thread/ISA sweep використовувати тільки options, підтверджені поточним help, і не перевищувати cpuset.

Перший serving baseline формується з exact checkpoint: offload, no speculation, default compute precision, declared context, fixed expert slots із fit ledger. Повний command отримується у 02a/05a; чисел expert slots до metadata і hardware вимірів тут немає навмисно.

Sweep budget: threads = unique sorted {1,2,4,P-2,P-1,P} intersect [1,P], де P — доступні physical cores; SMT окремо тільки після цього. Max-fetch grid = {auto,0,1,2,min(4,actual active expert count)}, unsupported відкинути. PLE cache grid у CONTRACTS.md. MTP d=0/1/2/3 окремо від lookup d=0/1/2/4. Graph=off/on. Context=4K/16K/32K, optional64K. Не виконувати повний Cartesian product: stage-by-stage one-factor sweep і combined retest переможців.

Кожний run має timeout, власний PID, stdout/stderr log, readiness deadline і guaranteed cleanup у finally. Port визначається через bind OS, а не fixed busy port. Результат не записується success до completion. Поточний model/process не перезапускається імпортером.

Порядок ABBA чергувати між candidates, не міряти всі baseline лише до нагрівання GPU. Tokenized corpus фіксувати один раз, hash у report. Не виконувати generated code з model output без окремого test sandbox.
