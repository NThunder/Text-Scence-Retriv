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

"Анализ влияния детализации текстовых описаний на качество мультимодального поиска в сценах автономного вождения"

Финальная сводная таблица
L0→L0 (честный baseline, короткие описания, мягкая релевантность)
Модель	R@1	R@5	R@10	MRR	Median Rank
CLIP ViT-L/14	0.294	0.558	0.731	0.412	4.0
SigLIP	0.277	0.624	0.731	0.446	3.0
EVA-CLIP-B/16	0.442	0.697	0.709	0.536	2.0
EVA-CLIP-L/14-336	0.498	0.733	0.816	0.611	2.0
LLM2CLIP (zero-shot)	0.556	0.767	0.922	0.652	1.0
LLM2CLIP + LoRA (image only)	0.695	0.820	0.923	0.750	1.0
Full LoRA (image+text)	0.738	0.833	0.914	0.785	1.0
Utonia + weighted fusion	0.738	0.923	0.966	0.825	1.0
GME zero-shot	0.721	0.777	0.777	0.754	1.0
GME fine-tuned	0.751	0.850	0.991	0.807	1.0

Смешанный режим (релевантность L0, текст варьируется — только GME)
Уровень текста	Zero-shot R@1	Fine-tuned R@1	Cosine sim текстов
L0 attr	0.721	0.751	0.449
L1 attr+motion	0.661	0.850	0.484
L2 attr+map	0.678	0.773	0.469
L3 attr+motion+map	0.721	0.807	0.498
L4 full	0.640	0.785	0.625

Финальная таблица L3→L3 (строгий режим, полная)
Модель	R@1	R@5	R@10	MRR	Median Rank	Avg rel. set
LLM2CLIP (no LoRA)	0.343	0.519	0.579	0.429	4.0	17.4
LLM2CLIP + LoRA (image only)	0.382	0.614	0.803	0.493	3.0	17.4
GME zero-shot	0.416	0.579	0.695	0.513	—	17.4
Full LoRA (image+text)	0.502	0.670	0.803	0.581	1.0	17.4
Utonia + weighted fusion	0.498	0.712	0.803	0.591	2.0	17.4
GME fine-tuned	0.575	0.743	0.863	0.660	1.0	17.4

Полная таблица результатов (GME zero-shot)
Режим	Уровень запроса	Уровень релевантности	R@1	R@5	MRR	Avg rel. set	Cosine sim
Смешанный	L0	L0	0.721	0.777	0.754	36.7	0.449
Смешанный	L1	L0	0.661	0.777	0.719	36.7	0.484
Смешанный	L2	L0	0.678	0.850	0.763	36.7	0.469
Смешанный	L3	L0	0.721	0.850	0.784	36.7	0.498
Смешанный	L4	L0	0.640	0.712	0.687	36.7	0.625
Строгий	L3	L3	0.416	0.579	0.513	17.4	0.498
Строгий	L4	L4	0.155	0.283	0.229	7.7	0.625

Полные результаты — обе модели, все уровни
Режим "смешанный" (фиксированная релевантность L0, меняется только текст запроса)
Уровень	Пример описания	Zero-shot R@1	Fine-tuned R@1	Δ от FT
L0 attr	"A blue car."	0.721	0.751	+0.030
L1 attr+motion	"A blue car is moving quickly."	0.661	0.850	+0.189
L2 attr+map	"A blue car in the stop lane."	0.678	0.773	+0.095
L3 attr+motion+map	"A blue car is moving quickly in the stop lane."	0.721	0.807	+0.086
L4 full	+ depth + loc + relation	0.640	0.785	+0.145
Режим "строгий" (релевантность = тот же уровень)
Уровень	Rel. set	Zero-shot R@1	Fine-tuned R@1
L0→L0	26.7	0.721	0.751
L3→L3	9.4	0.416	0.575
L4→L4	5.7	0.155	0.240

Шаг 0: Генерация новых caption-файлов с min_lidar_points=5
# L0
python retriv/build_camera_aware_captions.py \
    --level attr --min_lidar_points 5 \
    --output_path ./retriv/camera_aware_captions_L0_p5.json
# L1
python retriv/build_camera_aware_captions.py \
    --level attr_motion --min_lidar_points 5 \
    --output_path ./retriv/camera_aware_captions_L1_p5.json
# L2
python retriv/build_camera_aware_captions.py \
    --level attr_map --min_lidar_points 5 \
    --output_path ./retriv/camera_aware_captions_L2_p5.json
# L3
python retriv/build_camera_aware_captions.py \
    --level attr_motion_map --min_lidar_points 5 \
    --output_path ./retriv/camera_aware_captions_L3_p5.json
# L4
python retriv/build_camera_aware_captions.py \
    --level full --min_lidar_points 5 \
    --output_path ./retriv/camera_aware_captions_L4_p5.json



Константы (задать один раз)
L0=./retriv/camera_aware_captions_L0_p5.json
L3=./retriv/camera_aware_captions_L3_p5.json
L1=./retriv/camera_aware_captions_L1_p5.json
L2=./retriv/camera_aware_captions_L2_p5.json
L4=./retriv/camera_aware_captions_L4_p5.json
FLAGS="--disable_temporal_relevance --relevance_mode all"
GPU="CUDA_VISIBLE_DEVICES=2"

# ── EVA-CLIP-B/16 (768d) ──────────────────────────────────────
python retriv/encode_camera_aware_captions_evaclip.py \
    --camera_captions_path $L0 \
    --output_path ./text_embs_evaclipB16_L0_p5.pth
python retriv/encode_nuscenes_images_evaclip.py \
    --output_path ./img_embs_evaclipB16_L0_p5.pth
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path ./text_embs_evaclipB16_L0_p5.pth \
    --image_emb_path ./img_embs_evaclipB16_L0_p5.pth \
    $FLAGS
# ── EVA-CLIP-L/14-336 (1024d) ────────────────────────────────
python retriv/encode_camera_aware_captions_evaclip_l14_336.py \
    --camera_captions_path $L0 \
    --output_path ./text_embs_evaclipL14_L0_p5.pth
python retriv/encode_nuscenes_images_evaclip_l14_336.py \
    --output_path ./img_embs_evaclipL14_L0_p5.pth
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path ./text_embs_evaclipL14_L0_p5.pth \
    --image_emb_path ./img_embs_evaclipL14_L0_p5.pth \
    $FLAGS
# ── SigLIP (768d) ─────────────────────────────────────────────
python retriv/encode_camera_aware_captions_siglip.py \
    --camera_captions_path $L0 \
    --output_path ./text_embs_siglip_L0_p5.pth
python retriv/encode_nuscenes_images_siglip.py \
    --output_path ./img_embs_siglip_L0_p5.pth
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path ./text_embs_siglip_L0_p5.pth \
    --image_emb_path ./img_embs_siglip_L0_p5.pth \
    $FLAGS
# ── LLM2CLIP zero-shot (1280d) ────────────────────────────────
python retriv/encode_camera_aware_captions_llm2clip.py \
    --camera_captions_path $L0 \
    --output_path ./text_embs_llm2clip_L0_p5.pth
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
    --output_path ./img_embs_llm2clip_L0_p5.pth
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path ./text_embs_llm2clip_L0_p5.pth \
    --image_emb_path ./img_embs_llm2clip_L0_p5.pth \
    $FLAGS


tab:lora — LLM2CLIP + LoRA, L0→L0
# Переобучение LoRA image only
python retriv/train_lora_image_encoder.py \
    --text_emb_path ./text_embs_llm2clip_L0_p5.pth \
    --output_lora_dir ./lora_image_only_p5
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
    --lora_path ./lora_image_only_p5 \
    --output_path ./img_embs_lora_image_L0_p5.pth
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path ./text_embs_llm2clip_L0_p5.pth \
    --image_emb_path ./img_embs_lora_image_L0_p5.pth \
    $FLAGS
# Переобучение Full LoRA (image+text)
python retriv/train_joint_lora_encoders.py \
    --camera_captions_path $L3 \
    --output_lora_text ./lora_text_p5 \
    --output_lora_image ./lora_image_p5
python retriv/encode_camera_aware_captions_llm2clip.py \
    --lora_path ./lora_text_p5 \
    --camera_captions_path $L0 \
    --output_path ./text_embs_fullora_L0_p5.pth
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
    --lora_path ./lora_image_p5 \
    --output_path ./img_embs_fullora_L0_p5.pth
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path ./text_embs_fullora_L0_p5.pth \
    --image_emb_path ./img_embs_fullora_L0_p5.pth \
    $FLAGS
tab:utonia — механизмы слияния, L3→L3
# Переобучение Utonia на L3_p5
python retriv/train_with_utonia.py \
    --fusion_type weighted \
    --camera_captions_path $L3 \
    --output_logs ./training_logs_utonia_p5
# eval L3→L3
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L3 \
    --text_emb_path ./training_logs_utonia_p5/embeddings/epoch_best/text_embeddings.pth \
    --image_emb_path ./training_logs_utonia_p5/embeddings/epoch_best/fused_embeddings.pth \
    $FLAGS
tab:utonia_weighted_modes — Utonia L0→L0 и L3→L3
# L0→L0
python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path ./training_logs_utonia_p5/embeddings/epoch_best/text_embeddings.pth \
    --image_emb_path ./training_logs_utonia_p5/embeddings/epoch_best/fused_embeddings.pth \
    $FLAGS
# L3→L3 (уже выше)
tab:mixed — GME, смешанный режим L1-L4 → L0
# GME zero-shot
for LEVEL in L0 L1 L2 L3 L4; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name NCSOFT/GME-VARCO-VISION-Embedding \
    --camera_captions_path $CAPS \
    --relevance_captions_path $L0 \
    --output_logs ./logs_gme_zero_${LEVEL}_to_L0_p5 \
    $FLAGS
done
# GME fine-tuned на L3_p5 (вариант A)
python retriv/finetune_gme_nuscenes.py \
    --camera_captions_path $L3 \
    --output_dir ./gme_finetuned_L3_p5 \
    --batch_size 8 --lr 2e-4 --epochs 10 --use_lora
for LEVEL in L0 L1 L2 L3 L4; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L3_p5/best_model \
    --camera_captions_path $CAPS \
    --relevance_captions_path $L0 \
    --output_logs ./logs_gme_ft_L3_${LEVEL}_to_L0_p5 \
    $FLAGS
done
# GME fine-tuned на L0_p5 (вариант B)
python retriv/finetune_gme_nuscenes.py \
    --camera_captions_path $L0 \
    --output_dir ./gme_finetuned_L0_p5 \
    --batch_size 8 --lr 2e-4 --epochs 10 --use_lora
for LEVEL in L0 L1 L2 L3 L4; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L0_p5/best_model \
    --camera_captions_path $CAPS \
    --relevance_captions_path $L0 \
    --output_logs ./logs_gme_ft_L0_${LEVEL}_to_L0_p5 \
    $FLAGS
done
tab:strict — строгий режим L3→L3 и L4→L4
for LEVEL in L3 L4; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name NCSOFT/GME-VARCO-VISION-Embedding \
    --camera_captions_path $CAPS \
    --output_logs ./logs_gme_zero_${LEVEL}_strict_p5 \
    $FLAGS
  $GPU python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L3_p5/best_model \
    --camera_captions_path $CAPS \
    --output_logs ./logs_gme_ft_L3_${LEVEL}_strict_p5 \
    $FLAGS
done

