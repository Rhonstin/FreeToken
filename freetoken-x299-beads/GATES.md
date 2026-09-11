# Gates і умови зупинки

1. CONTRACT: усі inputs, invariants, negative cases і commands визначені; baseline перевірений. Незавершена a блокує b.
2. CORRECTNESS: усі required cases pass. Нуль необґрунтованих skips. No CUDA hardware => blocked GPU acceptance, CPU-only contract роботу можна продовжити.
3. MEMORY: declared maximum context проходить prefill і sustained decode, RAM reserve лишається, OOM/sustained swapping немає.
4. PERFORMANCE: paired median gain >=5%, і 95% bootstrap confidence interval нижньої межі paired speed ratio >1.0; p95 inter-token/chunk metric regression <=5% на declared workloads. Якщо n=3 недостатньо, до 5, потім до 9 paired runs; невизначеність після цього => inconclusive/no default change. Це policy приймання, не обіцянка speedup.
5. QUALITY: reference fixtures і fixed model corpus. NLL unavailable явно blocked; не заявляти quantitative quality-neutral precision conversion без відповідного тесту.
6. ROLLBACK: default/off повертає baseline behavior; повторний launch і smoke проходять. Не використовувати git reset --hard для rollback: isolated branch/worktree або вимкнення опції.
7. HANDOFF: raw paths, SHA, commands, flags, case IDs, failures/skips, actual Beads status. Імпортер або агент не закриває task за наявність коду.

OOM: завершити лише власний benchmark subprocess, зберегти logs, відхилити candidate, зменшити наступний candidate у bounded grid. Не kill-all GPU processes. Два однакові correctness failures => припинити speed tuning цієї гілки й створити bug/blocker. Deadlock timeout зафіксувати до run; не перезапускати нескінченно. Baseline drift => rebase contract/evidence, не переносити старі green результати.

No-change є успішним завершенням дослідження лише якщо baseline і candidate реально виміряні або trace доводить відсутність очікуваного bottleneck. «Не вистачило часу/доступу» => blocked, не no-change. Критерії успіху не змінювати після перегляду результатів.
