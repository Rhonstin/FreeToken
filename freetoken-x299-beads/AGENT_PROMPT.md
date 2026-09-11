Версія 2: обов’язково прочитай CONTRACTS.md, GATES.md, RUNBOOK.md, CPU_BRANCHES.md і MIGRATION.md. Task 01 тепер 01a/01b тощо; a блокує b. Не закривай a за порожній шаблон: потрібні actual fixtures/commands.

Прочитай AGENTS.md, CONTRIBUTING.md та .agents/skills/beads/SKILL.md у поточному FreeToken. Виконай bd prime. Цей пакет є окремим optimization epic, не продовженням codec implementation.

Після успішного імпорту читай freetoken-x299-beads/PLAN.uk.md і BENCHMARK_CONTRACT.md. Використовуй bd ready, bd show та bd update <id> --claim. Статуси веди лише в Beads; task markdown — вхідні acceptance descriptions. Не створюй паралельний TODO/MEMORY tracker.

Ціль: максимальна виміряна швидкість single-request committed generation Qwen3.8-Flash-Next на RTX3090, PCIe3 x16, 104 GB DDR4-2133, X299/i7. Exact CPU/channels/storage/checkpoint спочатку зібрати. Не проси користувача вибирати технічні defaults, які можна визначити read-only діагностикою. Якщо target server недоступний — підготуй відтворювані команди і познач hardware tasks blocked, не запускай їх на іншій машині як target benchmark.

У fork уже є KV quantization, MTP, adaptive depth, prompt lookup та paired benchmark. Перевір їх actual стан і Beads IDs; не створюй ці компоненти вдруге. Спочатку best-known settings і profiling, потім тільки зміни з доказами. Tasks 09–14/16–19 можуть завершитися no-change ADR лише після реальних вимірів. Це допустимий результат експерименту, не реалізоване прискорення.

Не змінюй router/top-k/семантику моделі заради throughput. Не вважай accepted drafts output tokens, не вмикай MTP/lookup разом. Для всіх кешів збережи graph/cancellation/rebuild/rollback correctness. Не pin-ити всю RAM без peak budget. Не вмикай непідтримувані ISA чи native FP8/Blackwell kernels на 3090.

До bd close приклади raw evidence, точні команди/revisions, changed paths, acceptance outcomes і limitations. Дотримуйся conservative repo policy: без auto commit/push/PR/Dolt remote sync. Наприкінці — стан задач і наступна ready task.
