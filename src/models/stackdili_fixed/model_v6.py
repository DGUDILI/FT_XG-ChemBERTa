import os
import shutil
import subprocess
import numpy as np
import pandas as pd

from models.stackdili_fixed.ft.ft_v6.ft_v6 import FTv6
from models.stackdili_fixed.stacking.base import BaseStacking


class ModelV6:
    """FTv6 (FP 선택 + ChemBERTa + Cross-Attention) → Stacking 파이프라인.

    흐름:
        Feature_raw_rdkit.csv 복원
        → (선택) 데이터 정제
        → FTv6.fit()   : FTv4.5 FP 선택 + 인코더 학습
        → FTv6.transform() : 16-dim 피처 행렬 생성
        → Stacking.fit() / evaluate()
    """

    def __init__(
        self,
        stacking: BaseStacking,
        stacking_version: str = "unknown",
        ft_version: str = "f6",
    ):
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.stacking         = stacking
        self.stacking_version = stacking_version
        self.ft_version       = ft_version
        self.ft               = FTv6()

    # ------------------------------------------------------------------

    def _restore_features(self, features_path: str) -> None:
        features_dir = os.path.dirname(features_path)
        raw_path = os.path.join(features_dir, self.ft.feature_raw_csv)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"[오류] FTv6 전용 피처 파일이 없습니다: {raw_path}\n"
                f"  먼저 './run.sh add-features' 를 실행하세요."
            )
        shutil.copy2(raw_path, features_path)
        print(f"[원본 복원] {self.ft.feature_raw_csv} → Feature.csv")

    def _build_save_dir(self, clean: bool) -> str:
        dir_name = f"stacking_{self.stacking_version}_ft_{self.ft_version}"
        if clean:
            dir_name += "_clean"
        return os.path.join(
            self.project_root, "src", "models", "stackdili_fixed", "Model", dir_name
        )

    # ------------------------------------------------------------------

    def run(self, clean: bool = False) -> None:
        features_path = os.path.join(
            self.project_root, "src", "features", "Feature.csv"
        )

        # 매 실행 시 RDKit 원본 복원
        self._restore_features(features_path)

        # 1. 데이터 정제 (선택)
        if clean:
            print("[1/3] 데이터 정제")
            clean_script = os.path.join(
                self.project_root, "src", "preprocessing", "make_clean_data.py"
            )
            subprocess.run(["python", clean_script], check=True, text=True)
            cleaned_csv = os.path.join(
                self.project_root, "src", "features", "Feature_cleaned.csv"
            )
            shutil.copy2(cleaned_csv, features_path)

        # 2. 데이터 로드
        raw = pd.read_csv(features_path)

        train = raw[raw["ref"] != "DILIrank"].reset_index(drop=True)
        test  = raw[raw["ref"] == "DILIrank"].reset_index(drop=True)

        X_train      = train.drop(["SMILES", "Label", "ref"], axis=1)
        y_train      = train["Label"]
        smiles_train = train["SMILES"]

        X_test       = test.drop(["SMILES", "Label", "ref"], axis=1)
        y_test       = test["Label"].values
        smiles_test  = test["SMILES"]

        # 3. FTv6 학습
        print("[2/3] FTv6 인코더 학습")
        self.ft.fit(X_train, y_train, smiles_train)

        # 4. 16-dim 피처 변환
        cols = [f"dim_{i}" for i in range(FTv6.HIDDEN_DIM)]
        X_train_16 = pd.DataFrame(
            self.ft.transform(X_train, smiles_train), columns=cols
        )
        X_test_16 = pd.DataFrame(
            self.ft.transform(X_test, smiles_test), columns=cols
        )

        # 5. Stacking 학습 및 평가
        print("[3/3] 스태킹 학습 및 평가")
        save_dir = self._build_save_dir(clean)
        os.makedirs(save_dir, exist_ok=True)
        self.stacking.fit(X_train_16, y_train.values, X_test_16, y_test, save_dir)
        self.stacking.evaluate(X_test_16, y_test, save_dir)

        print(f"Stacking {self.stacking_version}  |  FTv6  |  clean={clean}")

    def predict(self, _):
        return None
