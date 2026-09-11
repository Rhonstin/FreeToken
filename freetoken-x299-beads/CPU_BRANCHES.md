# X299 i7: автоматичний вибір гілки

X299/LGA2066 підтримує кілька i7. Наприклад, i7-7740X має 2 memory channels і AVX2, а i7-7800X/7820X підтримують AVX-512 та до 4 memory channels. Це capabilities SKU, не доказ реально активних каналів у вашій конфігурації. Заявлені 104 GB не дають надійного визначення exact SKU.

| Спостереження | Дія агента |
|---|---|
| SKU визначений, avx2 є, avx512f немає | AVX2 branch; жодного AVX512 запуску |
| avx512f є | Перевірити всі flags, потрібні конкретному compiled kernel, потім A/B AVX2/AVX512 зі clocks |
| SKU unknown, flags доступні | Generic flags-gated implementation; CPU-specific preset blocked |
| DIMM/channel evidence incomplete | Залишити channels unknown; використовувати measured bandwidth, не теоретичне quad-channel |
| CPUID flags підтримують ISA, extension зібрано без неї | Compiled capability gate відхиляє path |
| SKU з документації не узгоджується з runtime | Runtime evidence і isolation/VM діагностика; не force ISA |

Початковий thread grid задається у RUNBOOK.md, не фіксується як «8 потоків для будь-якого X299». Hardware collector не читає serial numbers і не змінює clocks/BIOS.

Офіційні джерела Intel, перевірені при підготовці v2:

- https://www.intel.com/content/www/us/en/products/sku/121499/intel-core-i77740x-xseries-processor-8m-cache-up-to-4-50-ghz/specifications.html
- https://www.intel.com/content/www/us/en/products/compare.html?productIds=%2F126684%2C123589%2C123767
- https://www.intel.com/content/www/us/en/support/articles/000006778/processors.html
