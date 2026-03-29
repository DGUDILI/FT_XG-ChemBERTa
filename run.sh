#!/usr/bin/env bash

CMD=${1}

case "$CMD" in
  build)
    docker compose build
    ;;
  shell)
    docker compose run --rm ml bash
    ;;
  run)
    STACKING="s1"
    FT=""
    CLEAN=""

    # 접두사(s=stacking, f=ft)로 인자 구분 — 순서 무관
    for arg in "${@:2}"; do
      case "$arg" in
        s*)  STACKING="$arg" ;;
        f*)  FT="$arg" ;;
        clean) CLEAN="clean" ;;
      esac
    done

    FT_ARG=""
    CLEAN_ARG=""
    if [ -n "$FT" ]; then
      FT_ARG="--ft=$FT"
    fi
    if [ "$CLEAN" = "clean" ]; then
      CLEAN_ARG="--clean"
    fi
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/train.py --stacking="$STACKING" $FT_ARG $CLEAN_ARG
    ;;
  env-test)
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/env_test.py
    ;;
  add-features)
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/features/add_rdkit_features.py
    ;;
  *)
    echo "Usage: ./run.sh {build|shell|run [...options]|env-test|add-features}"
    echo "  stacking: s0 | s0.5 | s1 | s3   (기본값: s1, 접두사 s)"
    echo "  ft:       f0 | f4.5 (생략 가능, 접두사 f)"
    echo "  clean:    'clean' 입력 시 Train-Test 중복 제거 실행"
    echo "  * 순서 무관"
    echo ""
    echo "  예시: ./run.sh run                  # Stacking s1, FT 없음"
    echo "        ./run.sh run s1               # Stacking s1, FT 없음"
    echo "        ./run.sh run s1 f4.5          # Stacking s1 + FT f4.5"
    echo "        ./run.sh run f0 s1            # 순서 바꿔도 동일"
    echo "        ./run.sh run s1 f4.5 clean    # Stacking s1 + FT f4.5 + 정제"
    echo "        ./run.sh run clean s1         # Stacking s1 + 정제, FT 없음"
    echo "        ./run.sh add-features         # RDKit feature 추가 (최초 1회)"
    exit 1
    ;;
esac
