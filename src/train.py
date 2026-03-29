import argparse
from registry import FT_REGISTRY, STACKING_REGISTRY, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacking", default="v1", choices=list(STACKING_REGISTRY),
                        help="Stacking 버전 (예: s1, s3)")
    parser.add_argument("--ft", default=None, choices=list(FT_REGISTRY.keys()),
                        help="FT 버전 (예: f0, f4.5). 생략하면 FT 없이 실행")
    parser.add_argument("--clean", action="store_true",
                        help="Train-Test 중복 제거 데이터 정제 실행")
    args = parser.parse_args()

    model = build_model(stacking_version=args.stacking, ft_version=args.ft)
    model.run(clean=args.clean)


if __name__ == "__main__":
    main()
