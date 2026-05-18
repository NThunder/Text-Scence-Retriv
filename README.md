```markdown
# Документация экспериментов по мультимодальному поиску (Text-Scene Retrieval)

В рамках проекта разработаны и протестированы несколько подходов для совместного обучения представлений **изображений, текста и облаков точек** на основе датасета nuScenes. Цель — построить единое эмбеддинг-пространство для задач поиска (retrieval) и семантического выравнивания.

## Структура репозитория (добавляемые файлы)

```
M3Net/
├── retriv/
│   ├── train_three_lora_encoders.py          # обучение трёх энкодеров (изображение, текст, облако)
│   ├── train_with_utonia.py                  # обучение с энкодером Utonia (PTv3) + fusion
│   ├── finetune_gme_nuscenes.py              # дообучение GME-VARCO-VISION-Embedding
│   ├── validate_gme_vlm.py                   # валидация GME модели
│   ├── eval_three_modalities.py              # метрики для трёх энкодеров
│   ├── eval_clip_baseline_per_camera.py      # классическая оценка text↔image
│   ├── encode_camera_aware_captions_llm2clip.py
│   ├── encode_nuscenes_images_evaclip_l14_336_hf.py
│   ├── encode_nuscenes_pointclouds_*.py
│   ├── build_camera_aware_captions.py
│   └── debug_camera_aware_samples.py
├── camera_aware_captions_motion_map.json     # текстовые описания сцен (атрибуты)
├── camera_aware_captions_short.json          # упрощённые описания для быстрых экспериментов
├── requirements.txt
├── environment.yml                           (для Utonia)
└── README.md
```

**Файлы для Git**: все скрипты в `retriv/`, JSON‑файлы с captions, `README.md`, `requirements.txt`, `.gitignore`.  
**Игнорировать**: `data/`, `*.pth`, `*.pt`, `logs/`, `checkpoints/`, `embeddings/`, `__pycache__/`, `.mlspace/`.

---

## 0. Подготовка данных и окружения

### Датасет nuScenes

- Скачать полную версию `v1.0-trainval` с [официального сайта](https://www.nuscenes.org/download).
- Путь по умолчанию: `/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval` (можно переопределить аргументом `--dataroot`).

### Генерация описаний

```bash
python3 retriv/build_camera_aware_captions.py --output_path ./camera_aware_captions_motion_map.json
```

Скрипт использует аннотации из `final_caption_bbox_token.json` (1.5 GB) и формирует для каждого кадра текст вида:  
`"The camera view contains: car; pedestrian; road"`.

### Окружение

Для воспроизведения базового окружения:

```bash
conda env create -f retriv/environment.yaml
conda activate gbuhtuev_m3net
```

Для экспериментов с **Utonia** (PTv3) требуется отдельное окружение (см. инструкции в репозитории Utonia).  
Для **GME-VARCO-VISION-Embedding** рекомендуется окружение:

```bash
conda create -n gme_finetune python=3.10 -y
conda activate gme_finetune
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate qwen-vl-utils peft nuscenes-devkit tensorboard matplotlib tqdm
```

---

## 1. Визуализация и отладка (опционально)

```bash
python3 retriv/debug_camera_aware_samples.py
```

Создаёт HTML‑отчёт с изображениями и текстовыми описаниями в папке `./debug_samples/`.

---

## 2. Обучение моделей

### 2.1. Совместное обучение трёх энкодеров (изображение, текст, облако точек)

**Скрипт:** `retriv/train_three_lora_encoders.py`  
**Используемые модели:**  
- Vision: `LLM2CLIP-Openai-L-14-336` + LoRA  
- Text: `LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned` + LoRA  
- Point: **PointTransformerV2P** (обучается с нуля)

```bash
python3 retriv/train_three_lora_encoders.py \
    --camera_captions_path ./camera_aware_captions_motion_map.json \
    --epochs 60 --lr 3e-4 --batch_size 6 --lora_r 16 \
    --output_lora_text ./lora_text_encoder_three_motion \
    --output_lora_image ./lora_image_encoder_three_motion \
    --output_point_model ./point_encoder_three_motion
```

### 2.2. Обучение с энкодером Utonia (PTv3) и слиянием (fusion)

**Скрипт:** `retriv/train_with_utonia.py`  
**Типы слияния:** `query` (по умолчанию), `weighted`, `moe`

```bash
python retriv/train_with_utonia.py \
    --fusion_type weighted \
    --epochs 60 \
    --camera_captions_path ./camera_aware_captions_motion_map.json \
    --output_logs ./training_logs_utonia
```

Лучшие результаты на валидации (Motion Map):
- **Weighted fusion**: R@1 = 0.4979, R@5 = 0.7339, R@10 = 0.8841, MRR = 0.6166  
- **MoE fusion**: R@1 = 0.4421

На коротких описаниях (`camera_aware_captions_short.json`) результаты выше:
- Weighted: R@1 = 0.7768, R@5 = 0.9013, R@10 = 0.9356

### 2.3. Дообучение GME-VARCO-VISION-Embedding

**Скрипт:** `retriv/finetune_gme_nuscenes.py`  
Модель специализирована для поиска изображений по тексту.

```bash
python retriv/finetune_gme_nuscenes.py \
    --dataroot /path/to/nuScenes \
    --camera_captions_path ./camera_aware_captions_motion_map.json \
    --output_dir ./gme_finetuned \
    --batch_size 8 --lr 2e-4 --epochs 10 --use_lora
```

**Результаты после дообучения (Motion Map):**  
R@1 = 0.5837, R@5 = 0.7253, R@10 = 0.8670, MRR = 0.6652  
На коротких описаниях: R@1 = 0.8069, R@5 = 0.9099, R@10 = 0.9442

---

## 3. Генерация эмбеддингов

### 3.1. Текстовые эмбеддинги (LLM2CLIP)

```bash
python3 retriv/encode_camera_aware_captions_llm2clip.py \
    --lora_path ./lora_text_encoder_three_motion \
    --camera_captions_path ./camera_aware_captions_motion_map.json \
    --output_path ./camera_text_embeddings_three_motion_lora.pth
```

### 3.2. Эмбеддинги изображений

```bash
python3 retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
    --lora_path ./lora_image_encoder_three_motion \
    --output_path ./camera_image_embeddings_three_motion_lora.pth
```

### 3.3. Эмбеддинги облаков точек

**Вариант с упрощённым энкодером (если spconv проблемный):**  
```bash
python3 retriv/encode_nuscenes_pointclouds_simple.py \
    --point_model_path ./point_encoder_three_motion/point_model.pth \
    --point_proj_path ./point_encoder_three_motion/point_proj.pth \
    --output_path ./camera_point_embeddings_three_motion_lora.pth
```

**Вариант с полноценным Utonia (требует spconv):**  
```bash
python3 retriv/encode_nuscenes_pointclouds_with_lora.py \
    --point_model_path ./point_encoder_three_motion/point_model.pth \
    --point_proj_path ./point_encoder_three_motion/point_proj.pth \
    --output_path ./camera_point_embeddings_three_motion_lora.pth \
    --batch_size 8
```

---

## 4. Оценка качества (метрики)

### 4.1. Классическая оценка text↔image (как в CLIP baseline)

```bash
python3 retriv/eval_clip_baseline_per_camera.py \
    --attributes_path ./camera_aware_captions_motion_map.json \
    --text_emb_path ./camera_text_embeddings_three_motion_lora.pth \
    --image_emb_path ./camera_image_embeddings_three_motion_lora.pth
```

**Результаты (с LoRA на тексте и изображении, motion_map):**  
R@1 = 0.5021, R@5 = 0.6695, R@10 = 0.8026, MRR = 0.5813, Median Rank = 1.0

### 4.2. Оценка трёх модальностей (text, image, point cloud)

```bash
python3 retriv/eval_three_modalities.py \
    --text_emb_path ./camera_text_embeddings_three_motion_lora.pth \
    --image_emb_path ./camera_image_embeddings_three_motion_lora.pth \
    --point_emb_path ./camera_point_embeddings_three_motion_lora.pth \
    --attributes_path ./camera_aware_captions_motion_map.json
```

Пример вывода (после 40 эпох):
```
Image → Point:  R@1 = 0.1674, R@5 = 0.3948, R@10 = 0.5622
Point → Image: R@1 = 0.2189, R@5 = 0.3605, R@10 = 0.5322
Text → Point:  R@1 = 0.0086, R@5 = 0.2532, R@10 = 0.3648
```

### 4.3. Оценка GME-VARCO-VISION-Embedding

```bash
python retriv/validate_gme_vlm.py \
    --camera_captions_path ./camera_aware_captions_short.json \
    --output_logs ./vlm_validation_logs \
    --batch_size 8 --temporal_window 10
```

**Результаты на коротких описаниях (без дообучения):**  
R@1 = 0.5777, R@5 = 0.8374, R@10 = 0.8495, MRR = 0.6838

---

## 5. Дополнительные эксперименты (история)

| Конфигурация                                      | R@1    | R@5    | R@10   | MRR    |
|--------------------------------------------------|--------|--------|--------|--------|
| Без LoRA (motion map)                            | 0.3433 | 0.5193 | 0.5794 | 0.4288 |
| Только LoRA на image (motion map)                | 0.3820 | 0.6137 | 0.8026 | 0.4927 |
| LoRA на image + text (motion map)                | 0.5021 | 0.6695 | 0.8026 | 0.5813 |
| Три энкодера (image+text+point)                  | 0.1674*| 0.3948*| 0.5622*| 0.2866* |
| Utonia + weighted fusion (motion map)            | 0.4979 | 0.7339 | 0.8841 | 0.6166 |
| GME-VARCO-VISION (short, без дообучения)         | 0.5777 | 0.8374 | 0.8495 | 0.6838 |
| GME-VARCO-VISION (short, fine-tuned)             | 0.8069 | 0.9099 | 0.9442 | 0.8606 |
|* – метрика Image→Point                           |        |        |        |        |

---

## 6. Заключение

- **Лучшие результаты поиска text→image** демонстрирует дообученный **GME-VARCO-VISION-Embedding** (R@1 до 0.81 на коротких описаниях).
- **Добавление облака точек** (три энкодера) улучшает взаимное выравнивание изображений и лидарных данных (Image→Point R@1 ~0.17), но text→point остаётся сложной задачей.
- **Слияние (fusion) эмбеддингов изображения и облака** даёт прирост в задачах Point→Image (R@1 до 0.22) и Image→Point (0.17), особенно при использовании механизмов attention или взвешенной суммы.
- Для промышленного применения рекомендуется **GME-VARCO-VISION-Embedding** – высокая точность и поддержка мультимодальных запросов.

Все скрипты готовы к запуску на 4 GPU H100 (или одной GPU). Параметры (batch size, learning rate, количество эпох) могут быть адаптированы под конкретные вычислительные ресурсы.
```