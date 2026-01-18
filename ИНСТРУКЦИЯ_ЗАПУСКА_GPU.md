# Подробная инструкция по запуску бенчмарка GigaChat3-10B на GPU

## Оглавление

1. [Требования к системе](#требования-к-системе)
2. [Подготовка системы](#подготовка-системы)
3. [Установка зависимостей](#установка-зависимостей)
4. [Скачивание модели](#скачивание-модели)
5. [Запуск vLLM сервера](#запуск-vllm-сервера)
6. [Запуск бенчмарка](#запуск-бенчмарка)
7. [Анализ результатов](#анализ-результатов)
8. [Оптимизация под разные GPU](#оптимизация-под-разные-gpu)
9. [Решение проблем](#решение-проблем)

---

## Требования к системе

### Минимальные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| **GPU** | NVIDIA с 24GB VRAM (RTX 3090/4090) | NVIDIA A100 40GB+ |
| **CPU** | 8 ядер | 16+ ядер |
| **RAM** | 32GB | 64GB+ |
| **Диск** | 50GB SSD | 100GB+ NVMe SSD |
| **CUDA** | 12.0 | 12.1+ |
| **Python** | 3.10 | 3.11 |
| **OS** | Ubuntu 20.04+ / RHEL 8+ | Ubuntu 22.04 |

### Проверка GPU

```bash
# Проверить наличие NVIDIA GPU
nvidia-smi

# Вывод должен показать GPU с CUDA 12.0+:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
# | N/A   32C    P0    52W / 400W |      0MiB / 40960MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# Проверить версию CUDA
nvcc --version

# Проверить драйвер
cat /proc/driver/nvidia/version
```

---

## Подготовка системы

### Шаг 1: Обновление системы

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# RHEL/CentOS
sudo dnf update -y
```

### Шаг 2: Установка системных зависимостей

```bash
# Ubuntu/Debian
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    wget \
    build-essential

# RHEL/CentOS
sudo dnf install -y \
    python3.11 \
    python3.11-devel \
    git \
    curl \
    wget \
    gcc \
    gcc-c++ \
    make
```

### Шаг 3: Клонирование репозитория

```bash
# Клонировать репозиторий
git clone https://github.com/YOUR_USERNAME/gigachat_swe_benchmark.git
cd gigachat_swe_benchmark
```

---

## Установка зависимостей

### Шаг 1: Создание виртуального окружения

```bash
# Создать виртуальное окружение
python3.11 -m venv venv

# Активировать окружение
source venv/bin/activate

# Обновить pip
pip install --upgrade pip setuptools wheel
```

### Шаг 2: Установка PyTorch с CUDA

```bash
# Установить PyTorch с поддержкой CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Проверить установку
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Шаг 3: Установка vLLM

```bash
# Установить vLLM (версия 0.6.0 или выше)
pip install vllm>=0.6.0

# Установить дополнительные зависимости
pip install huggingface_hub transformers accelerate
```

### Шаг 4: Установка mini-swe-agent

```bash
# Установить mini-swe-agent (если есть setup.py в корне)
pip install -e .

# Или установить напрямую
pip install mini-swe-agent
```

### Шаг 5: Установка зависимостей для аналитики

```bash
pip install pandas matplotlib seaborn pyyaml tqdm
```

### Полная команда установки (одной строкой)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install vllm>=0.6.0 huggingface_hub transformers accelerate && \
pip install pandas matplotlib seaborn pyyaml tqdm
```

---

## Скачивание модели

### Вариант 1: Автоматическое скачивание (рекомендуется)

```bash
# Скачать модель через Python
python3 -c "
from huggingface_hub import snapshot_download
import os

# Определить директорию кэша
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
print(f'Скачивание модели в: {cache_dir}')

# Скачать модель
model_path = snapshot_download(
    'ai-sage/GigaChat3-10B-A1.8B',
    resume_download=True,  # Продолжить при обрыве
    max_workers=4  # Параллельное скачивание
)
print(f'Модель скачана в: {model_path}')
"
```

### Вариант 2: Скачивание с указанием директории

```bash
# Задать свою директорию для модели
export HF_HOME=/path/to/your/models

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ai-sage/GigaChat3-10B-A1.8B')
"
```

### Вариант 3: Использование huggingface-cli

```bash
# Установить CLI
pip install huggingface_hub[cli]

# Скачать модель
huggingface-cli download ai-sage/GigaChat3-10B-A1.8B --local-dir ./models/gigachat-10b
```

### Проверка скачанной модели

```bash
# Проверить размер модели (~20GB)
du -sh ~/.cache/huggingface/hub/models--ai-sage--GigaChat3-10B-A1.8B/

# Или в указанной директории
du -sh ./models/gigachat-10b/
```

---

## Запуск vLLM сервера

### Вариант 1: Быстрый запуск (рекомендуется)

```bash
# Запустить сервер в фоновом режиме
python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --served-model-name gigachat-10b \
    --trust-remote-code \
    --enforce-eager \
    2>&1 | tee vllm_server.log &

# Запомнить PID
echo $! > vllm_server.pid
echo "vLLM сервер запущен с PID: $(cat vllm_server.pid)"
```

### Вариант 2: Использование скрипта start_vllm_server.py

```bash
# Запуск через скрипт с автоматической установкой зависимостей
python3 start_vllm_server.py \
    --model-id ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --install-deps \
    --wait-timeout 600
```

### Вариант 3: Запуск через tmux/screen (для долгих сессий)

```bash
# Создать tmux сессию
tmux new-session -d -s vllm

# Запустить сервер в tmux
tmux send-keys -t vllm "source venv/bin/activate && python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --served-model-name gigachat-10b \
    --trust-remote-code" Enter

# Подключиться к сессии для мониторинга
tmux attach -t vllm

# Отключиться: Ctrl+B, затем D
```

### Ожидание запуска сервера

```bash
# Ждать готовности сервера (2-5 минут в зависимости от GPU)
echo "Ожидание запуска vLLM сервера..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Сервер готов!"
        break
    fi
    echo "Попытка $i/60..."
    sleep 5
done

# Проверить статус
curl http://localhost:8000/health
# Ожидаемый ответ: {"status":"ok"}

# Проверить доступные модели
curl http://localhost:8000/v1/models
```

### Тест API

```bash
# Тестовый запрос к модели
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gigachat-10b",
        "messages": [{"role": "user", "content": "Привет! Как дела?"}],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

---

## Запуск бенчмарка

### Тестовый запуск (5 инстансов)

```bash
# Запустить на 5 инстансах для проверки
python3 run_swebench_local.py \
    --subset lite \
    --slice 0:5 \
    -o test_results

# Время выполнения: ~10-30 минут
```

### Запуск на dev split (23 инстанса)

```bash
# Полный dev split
python3 run_swebench_local.py \
    --subset lite \
    --split dev \
    -o gigachat_dev_results

# Время выполнения: ~1-3 часа
```

### Полный бенчмарк (300 инстансов, test split)

```bash
# Полный SWE-bench Lite test split
python3 run_swebench_local.py \
    --subset lite \
    --split test \
    --slice 0:300 \
    -o gigachat_test_results

# Время выполнения: ~12-48 часов (зависит от GPU)
```

### Параллельный запуск (несколько workers)

```bash
# Запуск с 2 параллельными workers
python3 run_swebench_local.py \
    --subset lite \
    --split test \
    --slice 0:300 \
    --workers 2 \
    -o gigachat_parallel_results

# Внимание: требует больше RAM и может привести к OOM на GPU
```

### Запуск конкретных инстансов

```bash
# Фильтр по regex (только Django инстансы)
python3 run_swebench_local.py \
    --subset lite \
    --filter "django__django" \
    -o django_results

# Конкретный срез инстансов
python3 run_swebench_local.py \
    --subset lite \
    --slice 50:100 \
    -o slice_50_100_results
```

### Мониторинг прогресса

```bash
# В отдельном терминале следить за логами
tail -f gigachat_test_results/benchmark.log

# Или смотреть статус инстансов
ls gigachat_test_results/*.traj.json | wc -l
```

---

## Анализ результатов

### Быстрый анализ

```bash
# Базовый анализ
python3 analyze_results.py ./gigachat_test_results
```

### Полный анализ с графиками

```bash
# Полный анализ со всеми отчётами
python3 analyze_results.py ./gigachat_test_results --all
```

### Результаты анализа

После выполнения в директории появятся:

```
gigachat_test_results/
├── preds.json                      # Патчи для SWE-bench evaluation
├── analysis_report.json            # Детальный JSON отчёт
├── results_detailed.csv            # CSV с результатами по каждому инстансу
├── results_summary.csv             # Сводка по репозиториям
└── plots/
    ├── exit_status_distribution.png  # Распределение статусов выхода
    ├── step_distribution.png         # Распределение шагов
    └── repo_performance.png          # Производительность по репозиториям
```

### Оценка результатов на SWE-bench

```bash
# Установить swe-bench для evaluation
pip install swebench

# Запустить evaluation (требует много ресурсов)
python -m swebench.harness.run_evaluation \
    --predictions_path ./gigachat_test_results/preds.json \
    --swe_bench_tasks princeton-nlp/SWE-bench_Lite \
    --log_dir ./evaluation_logs \
    --testbed /tmp/swe_testbed \
    --skip_existing \
    --timeout 1800 \
    --verbose
```

---

## Оптимизация под разные GPU

### NVIDIA A100 40GB (рекомендуется)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --served-model-name gigachat-10b \
    --trust-remote-code
```

### NVIDIA A100 80GB

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 65536 \      # Увеличенный контекст
    --gpu-memory-utilization 0.95 \
    --served-model-name gigachat-10b \
    --trust-remote-code
```

### NVIDIA RTX 4090 24GB

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 16384 \      # Уменьшенный контекст
    --gpu-memory-utilization 0.85 \
    --served-model-name gigachat-10b \
    --trust-remote-code \
    --enforce-eager              # Отключить CUDA graphs
```

### NVIDIA RTX 3090 24GB

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 8192 \       # Минимальный контекст
    --gpu-memory-utilization 0.80 \
    --dtype float16 \            # Явно указать float16
    --served-model-name gigachat-10b \
    --trust-remote-code \
    --enforce-eager
```

### Несколько GPU (Tensor Parallelism)

```bash
# Для 2x A100 40GB
python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2 \   # Использовать 2 GPU
    --served-model-name gigachat-10b \
    --trust-remote-code
```

### Обновление конфига для меньших GPU

В файле `gigachat_swebench_local.yaml` измените:

```yaml
model:
  model_kwargs:
    max_tokens: 1024    # Уменьшить с 2048 до 1024
```

---

## Решение проблем

### CUDA Out of Memory

**Симптом:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Решения:**
1. Уменьшить `--max-model-len`:
   ```bash
   --max-model-len 16384  # вместо 32768
   ```

2. Уменьшить `--gpu-memory-utilization`:
   ```bash
   --gpu-memory-utilization 0.80  # вместо 0.90
   ```

3. Добавить `--enforce-eager`:
   ```bash
   --enforce-eager  # Отключить CUDA graphs
   ```

4. Уменьшить `max_tokens` в конфиге

### vLLM не запускается

**Симптом:**
```
Error: No CUDA GPUs are available
```

**Решения:**
1. Проверить драйвер:
   ```bash
   nvidia-smi
   ```

2. Проверить CUDA:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. Переустановить PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Ошибка "Model not found"

**Симптом:**
```
Error: Model ai-sage/GigaChat3-10B-A1.8B not found
```

**Решения:**
1. Проверить скачивание модели:
   ```bash
   ls ~/.cache/huggingface/hub/models--ai-sage--GigaChat3-10B-A1.8B/
   ```

2. Скачать заново:
   ```bash
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ai-sage/GigaChat3-10B-A1.8B')"
   ```

### Ошибка "Context Window Exceeded"

**Симптом:**
```
Error: This model's maximum context length is 32768 tokens
```

**Решения:**
1. В `gigachat_swebench_local.yaml`:
   ```yaml
   model:
     model_kwargs:
       max_tokens: 1024
   ```

2. Уменьшить `--max-model-len` при запуске vLLM

### Сервер отвечает 503/504

**Симптом:**
```
HTTP 503 Service Unavailable
```

**Решения:**
1. Подождать загрузки модели (2-5 минут)

2. Проверить логи:
   ```bash
   tail -f vllm_server.log
   ```

3. Проверить GPU нагрузку:
   ```bash
   watch -n 1 nvidia-smi
   ```

### Медленная генерация

**Причины и решения:**
1. Включить CUDA graphs (убрать `--enforce-eager`)
2. Увеличить `--gpu-memory-utilization`
3. Использовать `--dtype bfloat16` вместо `float16`
4. Проверить thermal throttling: `nvidia-smi -q -d TEMPERATURE`

---

## Полный скрипт запуска

Создайте файл `run_benchmark.sh`:

```bash
#!/bin/bash
set -e

# Конфигурация
MODEL_ID="ai-sage/GigaChat3-10B-A1.8B"
PORT=8000
MAX_MODEL_LEN=32768
GPU_UTIL=0.90
OUTPUT_DIR="gigachat_results_$(date +%Y%m%d_%H%M%S)"

echo "=== GigaChat3-10B SWE-bench Benchmark ==="
echo "Дата: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Активация окружения
source venv/bin/activate

# Запуск vLLM сервера
echo "1. Запуск vLLM сервера..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --served-model-name gigachat-10b \
    --trust-remote-code \
    2>&1 | tee $OUTPUT_DIR/vllm.log &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Ожидание готовности
echo "2. Ожидание готовности сервера..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Сервер готов!"
        break
    fi
    sleep 5
done

# Запуск бенчмарка
echo "3. Запуск бенчмарка..."
python3 run_swebench_local.py \
    --subset lite \
    --split test \
    --slice 0:300 \
    -o $OUTPUT_DIR

# Анализ результатов
echo "4. Анализ результатов..."
python3 analyze_results.py $OUTPUT_DIR --all

# Остановка сервера
echo "5. Остановка vLLM сервера..."
kill $VLLM_PID

echo ""
echo "=== Бенчмарк завершён ==="
echo "Результаты: $OUTPUT_DIR/"
echo "Отчёт: $OUTPUT_DIR/analysis_report.json"
```

Запуск:
```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

---

## Дополнительные ресурсы

- [GigaChat3-10B-A1.8B на HuggingFace](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)
- [SWE-bench официальный сайт](https://www.swebench.com/)
- [mini-swe-agent GitHub](https://github.com/klieret/mini-swe-agent)
- [vLLM документация](https://docs.vllm.ai/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
