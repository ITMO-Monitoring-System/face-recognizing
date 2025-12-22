# Face Recognition Service - Документация

## Назначение сервиса
Система для распознавания лиц. Сервис может:
- Сохранять эмбеддинги лиц студентов (по лекциям)
- Распознавать лица на изображениях в реальном времени
- Интегрироваться с внешними системами через RabbitMQ и HTTP API

## Быстрый старт

### Запуск проекта
```bash
docker-compose up -d
```

### Проверка работоспособности
После запуска выполните:

```bash
# Проверить состояние контейнеров
docker-compose ps

# Ожидаемый вывод:
# service           status   ports
# rabbitmq          running  0.0.0.0:5572->5672/tcp, 15672/tcp
# redis             running  0.0.0.0:6379->6379/tcp
# face-service      running  0.0.0.0:8180->8180/tcp

# Проверить здоровье сервиса
curl http://localhost:8180/health

# Ожидаемый ответ: {"ok": true}
```

## Компоненты системы

### Основные сервисы
| Сервис | Порт | Назначение |
|--------|------|------------|
| RabbitMQ | 5572 (AMQP), 15672 (UI) | Очереди для асинхронной обработки |
| Redis | 6379 | Хранение датасетов эмбеддингов |
| Face Service | 8180 | HTTP API для управления лекциями |

### Ключевые файлы
```
├── docker-compose.yml          # Конфигурация Docker
├── api.py                      # Основной HTTP API
├── face_service/core/          # Ядро распознавания
│   ├── recognize.py            # Логика распознавания
│   ├── retina_embending.py     # Извлечение эмбеддингов
│   └── persons_loader.py       # Загрузка данных
├── logic/dataset_store.py      # Redis хранилище
└── RabbitCheck/                # Тестовые утилиты
```

## 🔧 Тестирование функциональности

### 1. Тест через HTTP API
```bash
# Статус лекций
curl http://localhost:8180/api/lecture/status

# Запуск лекции
curl -X POST http://localhost:8180/api/lecture/start \
  -H "Content-Type: application/json" \
  -d '{
    "lecture_id": "lecture123",
    "in_amqp_url": "amqp://user:pass@rabbitmq:5672/",
    "in_queue": "faces_tasks",
    "threshold": 0.45
  }'

# Остановка лекции
curl -X POST http://localhost:8180/api/lecture/stop \
  -H "Content-Type: application/json" \
  -d '{"lecture_id": "lecture123"}'
```

### 2. Тест через RabbitMQ
```bash
# Отправить тестовое изображение
cd RabbitCheck
python send_task.py test_image.jpg test-request-1

# Проверить результаты
python read_result.py

# Запустить воркер
python worker.py
```

## API Endpoints

### Управление датасетами
- `POST /dataset` - Загрузить датасет эмбеддингов
- `DELETE /dataset/{lecture_id}` - Удалить датасет
- `POST /api/embedding` - Извлечь эмбеддинг из изображения

### Управление лекциями
- `POST /api/lecture/start` - Начать обработку лекции
- `POST /api/lecture/stop` - Остановить лекцию
- `GET /api/lecture/status` - Статус всех лекций
- `GET /api/lecture/last_result/{lecture_id}` - Последний результат

## Утилиты

### `tools/`
```bash
# Создание эмбеддингов
python tools/enroll.py

# Тестирование распознавания с JSON датасетом
python tools/recognize_json.py

# Тестирование с PostgreSQL
python tools/recognize_sql.py
```

### `RabbitCheck/`
- `send_task.py` - Отправка изображения в очередь
- `read_result.py` - Чтение результатов из очереди
- `worker.py` - Классический воркер для обработки задач

## Мониторинг

### RabbitMQ Management UI
- **URL**: http://localhost:15672
- **Логин**: user
- **Пароль**: pass

### Redis
```bash
# Проверить подключение
redis-cli -h localhost -p 6379 ping
# Ответ: PONG
```

## Устранение неполадок

### Проблема: Модель не загружается
**Решение**: Запустить предзагрузку модели:
```bash
docker-compose run --rm model-preload
```

### Проблема: RabbitMQ недоступен
**Решение**: Проверить порты и перезапустить:
```bash
docker-compose restart rabbitmq
```

### Проблема: Redis соединение разрывается
**Решение**: Проверить доступность памяти и увеличить таймауты:
```bash
docker-compose restart redis face-service
```

## Метрики работоспособности

1. **Время отклика API** (< 100 мс для health-check)
2. **Доступность RabbitMQ** (должен отвечать на ping)
3. **Доступность Redis** (должен отвечать на ping)
4. **Загрузка модели** (должна завершиться без ошибок)

## Интеграция с внешними системами

### Входные данные (от face-tracking)
- AMQP очередь с изображениями в base64
- Формат сообщения: `{"image_b64": "...", "threshold": 0.45}`

### Выходные данные (для backend)
- AMQP очередь с результатами
- Формат: `{"lecture_id": "...", "person_id": "..."}`

### HTTP уведомления
- Автоматические уведомления о старте/остановке лекций
- Конфигурируются через переменные окружения