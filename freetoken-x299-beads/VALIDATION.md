# Перевірка пакета v2

Перевірено Python syntax, DAG 40 tasks, усі body paths, імпорт 1 epic +40 tasks через simulated bd CLI, повторний запуск без дублювання та відмову при старому plan_id. Це тест протоколу імпортера з тестовим CLI, не реальний Dolt/Beads integration test. Реальний bd і GPU runtime тут не запускали. Hardware collector перевірений синтаксично; на target hardware не запускався.
