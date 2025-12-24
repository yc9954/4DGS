#!/bin/bash
# 학습 상태 빠르게 확인하는 스크립트

LOG_FILE="/workspace/4DGS/data/outputs/sample_synthetic_test/model/training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 학습 로그 파일이 없습니다"
    exit 1
fi

# 마지막 진행 상황 추출
LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)

if echo "$LAST_LINE" | grep -q "Training progress"; then
    # 정보 추출
    PERCENT=$(echo "$LAST_LINE" | grep -oP '\d+%' | head -1)
    ITER=$(echo "$LAST_LINE" | grep -oP '\d+/\d+' | head -1)
    TIME_INFO=$(echo "$LAST_LINE" | grep -oP '\[\d+:\d+<.*?\]' | head -1)
    LOSS=$(echo "$LAST_LINE" | grep -oP 'Loss=[\d.]+' | cut -d= -f2)
    PSNR=$(echo "$LAST_LINE" | grep -oP 'psnr=[\d.]+' | cut -d= -f2)
    SPEED=$(echo "$LAST_LINE" | grep -oP '[\d.]+it/s' | head -1)
    
    echo "=========================================="
    echo "📊 학습 진행 상황"
    echo "=========================================="
    echo "진행률: $PERCENT"
    echo "Iterations: $ITER"
    echo "시간: $TIME_INFO"
    echo "속도: $SPEED"
    echo ""
    echo "📈 성능:"
    echo "  Loss: $LOSS"
    echo "  PSNR: $PSNR dB"
    echo ""
    
    # 로그 파일 수정 시간
    LAST_MOD=$(stat -c "%y" "$LOG_FILE" 2>/dev/null | cut -d'.' -f1)
    NOW=$(date '+%Y-%m-%d %H:%M:%S')
    echo "마지막 업데이트: $LAST_MOD"
else
    echo "⚠️  학습 진행 상황을 찾을 수 없습니다"
    echo "마지막 로그:"
    echo "$LAST_LINE"
fi

