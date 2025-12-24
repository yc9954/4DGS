#!/bin/bash
# 실시간 학습 모니터링 스크립트
# 사용법: ./monitor_training.sh

LOG_FILE="/workspace/4DGS/data/outputs/sample_synthetic_test/model/training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  로그 파일을 찾을 수 없습니다: $LOG_FILE"
    echo "학습이 시작되면 자동으로 표시됩니다..."
    echo ""
    # 파일이 생성될 때까지 대기
    while [ ! -f "$LOG_FILE" ]; do
        sleep 2
    done
fi

echo "=========================================="
echo "📊 4DGS 학습 실시간 모니터링"
echo "=========================================="
echo "로그 파일: $LOG_FILE"
echo "Ctrl+C를 눌러 종료하세요"
echo "=========================================="
echo ""

# 실시간으로 로그 파일 tail
tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    # Training progress 라인 강조 표시
    if echo "$line" | grep -q "Training progress"; then
        echo -e "\033[2K\r$line"  # 현재 라인 덮어쓰기
    else
        echo "$line"
    fi
done

