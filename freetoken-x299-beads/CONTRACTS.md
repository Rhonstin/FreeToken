# Обов’язкові контракти v2

Це нормативні вимоги для всіх 40 задач. Конкретні baseline paths/symbols беруться з task body; запропоновані нові interfaces нижче не видаються за вже наявний API.

## Evidence layout

Для кожного task key `01a`...`20b`: `evidence/x299/<key>/contract.json`, `commands.json`, `result.json`, `raw/`. Evidence files — докази, не task tracker. Статус лише в Beads.

`contract.json` обов’язково містить:

- `schema_version`: 2; `task_key`; actual `bead_id`;
- `baseline_git_sha`, `worktree_diff_sha256`, `checkpoint_revision`, `hardware_fingerprint_sha256`;
- `scope_files`: список дозволених змін; `symbols`: existing/proposed позначені окремо;
- `inputs`: кожне значення з source і unit; unknown = null + reason;
- `invariants`: перевірювані твердження, не «має працювати»;
- `cases`: ID, given, action, expected, assertion, required_hardware;
- `commands`: argv arrays, cwd, timeout_s, expected_exit, artifact path;
- `performance_gate`, `quality_gate`, `rollback_recipe`.

`result.json`: task_key, outcome (contract_ready/implemented/no_change/blocked), tested_sha, cases (id, pass/fail/skip, command_id, log_path), measurements (run_id, raw_path), failures, limitations, rollback_result. Missing cases — failure приймання. null не перетворюється на нуль.

## Memory contract

Рахувати unique tensor storages, не всі views; units — bytes, звіт GiB окремо. ledger rows: name, owner, device, dtype/codec, logical_shape, physical_shape, payload_bytes, scale_bytes, fixed_bytes, residency, lifetime, peak_phase. Shared weight не рахувати двічі. RAM reserve: початково max(8 GiB, 10% MemTotal); це policy, не факт про ОС. Candidate, який потребує reserve, відхиляється або має новий погоджений budget із доказами. MemAvailable нижче reserve чи sustained swap-in під steady decode => fail capacity gate. Не дозволяти cache size із від’ємного remaining budget.

## PLE row-cache proposed contract

Key включає immutable checkpoint identity + source file/extent identity + table/head ID + row ID. Entry immutable packed row bytes. Source identity не змінюється протягом run; modification виявлена => invalidation/error, не silent reuse. Конфігурація cache_bytes=0 означає bypass. Payload+metadata мають hard accounting cap. Спочатку кандидати 0, 256 MiB, 1 GiB, 2 GiB тільки якщо залишають RAM reserve; random-row control обов’язковий. Перед fill dedup, cache hits, batch-read misses, validate complete bytes, then publish entries. In-flight bytes не evict/reuse до completion. Negative cases: truncated read, duplicate row, wrong source, malicious collision, capacity менше однієї row, mixed hit/miss error. Unit oracle порівнює source bytes без GPU; GPU output parity окремо.

## Hardware-profile proposed contract

Fingerprint: CPU vendor/family/model/stepping і effective ISA, process affinity/threads, observed RAM channel evidence/configured speed, PCIe BDF/negotiated link, GPU model/SM, runtime versions, expert format/H/I/top-k, concurrency. Не включати приватні serials. Unknown required fields => unverified, не compatible. profile schema_version і compatibility matcher чиста функція. NaN/Inf/<=0 bandwidth => reject. Atomic temp+rename; частковий JSON не використовувати. Legacy профіль не мовчки підвищується до validated.

## Exact speculation invariant

Після commit accepted prefix однаковий у host input_ids, device token pool, KV page ownership, QSA pending ring, GDN/PLE recurrent state та counters. Proposed tokens не виходять у stream до verify. Після partial reject restore snapshot/journal і replay лише committed branch. Обов’язкові d=0/1/2/3, accept=0/partial/all, EOS у draft, max-length boundary, cancellation і repeated graph replay. Greedy parity — exact token IDs; stochastic перевіряти acceptance/residual algorithm на fixed small logits fixtures, не вимагати token-identical незалежні stochastic runs.

## Numerical gates

Codec/storage byte tests — exact. Index/page mappings і counters — exact. Floating kernel tests — існуючий reference tolerance конкретної операції, зафіксований до зміни; нова tolerance потребує error analysis і baseline oracle. Єдиного «cosine > X для всього» немає. Same precision/runtime scheduling зміна повинна пройти same reference tests; checkpoint/precision зміна запускає окрему quality campaign, не оголошується чистою runtime optimization.
