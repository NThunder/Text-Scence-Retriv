#!/bin/bash
# Ablation study: влияние порога лидарных точек на retrieval-метрики
# Уровень L0 (attr only), GME zero-shot, all mode, без temporal
# Все пороги оцениваются на одинаковом множестве пар (пересечение)

set -e


# ========== Шаг 1: Генерация caption-файлов для разных порогов ==========
for P in 0 1 20; do
    echo "Generating captions for min_lidar_points=$P..."
    python build_camera_aware_captions.py \
        --level attr \
        --min_lidar_points $P \
        --output_path ./camera_aware_captions_L0_p${P}.json
done

echo ""
echo "=== Caption files generated ==="
ls -lh camera_aware_captions_L0_p*.json
echo ""

# ========== Шаг 2: Вычисление пересечения пар ==========
echo "Computing pair intersection across thresholds..."
python compute_val_pair_intersection.py
echo ""

# ========== Шаг 3: Запуск валидации для каждого порога на одинаковых парах ==========
FILTER_PAIRS=./val_pairs_intersection.json

for P in 0 1 5 20; do
    CAPS=./camera_aware_captions_L0_p${P}.json
    OUTDIR=./logs_lidar_ablation_p${P}
    echo "============================================"
    echo "Running GME zero-shot with min_lidar_points=$P"
    echo "  Captions:    $CAPS"
    echo "  Filter:      $FILTER_PAIRS"
    echo "  Output:      $OUTDIR"
    echo "============================================"
    python validate_gme_vlm.py \
        --model_name NCSOFT/GME-VARCO-VISION-Embedding \
        --camera_captions_path $CAPS \
        --output_logs $OUTDIR \
        --disable_temporal_relevance \
        --relevance_mode all \
        --max_val_samples 150 \
        --filter_pairs $FILTER_PAIRS
    echo ""
done

# ========== Шаг 4: Сводная таблица ==========
echo ""
echo "======================================"
echo "  LIDAR ABLATION RESULTS (FIXED SET) "
echo "======================================"
echo ""

INT_PAIRS=$(python3 -c "import json; print(len(json.load(open('$FILTER_PAIRS'))))")
printf "Все порогов: %d пар (пересечение)\n" $INT_PAIRS
echo ""
printf "%-8s %-8s %-8s %-8s %-8s\n" "Porog" "R@1" "R@5" "R@10" "MRR"
printf "%-8s %-8s %-8s %-8s %-8s\n" "------" "----" "----" "-----" "----"

for P in 0 1 5 20; do
    RESULT=./logs_lidar_ablation_p${P}/results.json
    if [ -f "$RESULT" ]; then
        R1=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f\"{d['R@1']:.4f}\")")
        R5=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f\"{d['R@5']:.4f}\")")
        R10=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f\"{d['R@10']:.4f}\")")
        MRR=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f\"{d['MRR']:.4f}\")")
        printf "%-8s %-8s %-8s %-8s %-8s\n" "$P" "$R1" "$R5" "$R10" "$MRR"
    else
        printf "%-8s %-8s\n" "$P" "no results"
    fi
done
echo ""
echo "Done!"
