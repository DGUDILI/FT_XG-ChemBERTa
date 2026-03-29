import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Optional

from models.stackdili_fixed.ft.ft_v4_5 import FTv4_5
from models.stackdili_fixed.ft.ft_v6.chemberta import ChemBERTaEncoder
from models.stackdili_fixed.ft.ft_v6.cross_attention import CrossAttention


# ---------------------------------------------------------------------------
# 내부 PyTorch 모듈
# ---------------------------------------------------------------------------

class _FTv6Encoder(nn.Module):
    """FP_k → 16  /  raw ChemBERTa 768 → 16  /  CrossAttention → 16.

    ChemBERTa backbone은 외부에서 1회 pre-compute 후 raw 768-dim 텐서로 전달.
    학습 파라미터: fp_proj, chem_proj, cross_attn.
    """

    HIDDEN   = 16
    CHEM_DIM = 768  # ChemBERTa hidden_size

    def __init__(self, fp_dim: int, attn_mode: str = "fp_query"):
        """
        attn_mode:
            "fp_query"   — exp1: FP→Q,       ChemBERTa→K,V
            "chem_query" — exp2: ChemBERTa→Q, FP→K,V
            "bidirect"   — exp3: 양방향 concat → Linear(32, 16)
        """
        super().__init__()
        self.attn_mode  = attn_mode
        self.fp_proj    = nn.Linear(fp_dim,        self.HIDDEN)
        self.chem_proj  = nn.Linear(self.CHEM_DIM,  self.HIDDEN)
        self.cross_attn = CrossAttention(dim=self.HIDDEN, n_heads=4)
        # exp3 전용: 양방향 결과(32) → 16
        if attn_mode == "bidirect":
            self.fusion = nn.Linear(self.HIDDEN * 2, self.HIDDEN)

    def forward(self, fp: torch.Tensor, chem_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fp:       (B, fp_dim) — 선택된 FP 피처
            chem_raw: (B, 768)    — 캐싱된 ChemBERTa [CLS] 임베딩
        Returns:
            fused: (B, 16)
        """
        fp_16   = self.fp_proj(fp)           # (B, 16)
        chem_16 = self.chem_proj(chem_raw)   # (B, 16)

        if self.attn_mode == "fp_query":
            return self.cross_attn(fp_16, chem_16)          # FP→Q
        elif self.attn_mode == "chem_query":
            return self.cross_attn(chem_16, fp_16)          # ChemBERTa→Q
        else:  # bidirect
            out_fp   = self.cross_attn(fp_16,   chem_16)    # FP→Q
            out_chem = self.cross_attn(chem_16, fp_16)      # ChemBERTa→Q
            return self.fusion(torch.cat([out_fp, out_chem], dim=-1))  # (B, 16)


# ---------------------------------------------------------------------------
# 공개 클래스
# ---------------------------------------------------------------------------

class FTv6:
    """FTv4.5 FP 선택 → fp_proj 16-dim  +  ChemBERTa(캐시) → chem_proj 16-dim
    → Cross-Attention → 16-dim 피처 행렬.

    ChemBERTa 인코딩은 fit()/transform() 진입 시 1회만 실행 후 numpy로 캐싱.
    이후 학습 루프는 작은 linear layer만 반복 호출하므로 빠름.

    model_v6.py 전용 인터페이스:
        fit(X, y, smiles)     → FP 선택 + ChemBERTa 캐싱 + 인코더 학습
        transform(X, smiles)  → (n_samples, 16) ndarray 반환
    """

    feature_raw_csv = "Feature_raw_rdkit.csv"
    HIDDEN_DIM = 16

    def __init__(
        self,
        attn_mode:   str   = "fp_query",   # "fp_query" | "chem_query" | "bidirect"
        n_epochs:    int   = 50,
        lr:          float = 1e-3,
        batch_size:  int   = 32,
        random_seed: int   = 42,
        device:      Optional[str] = None,
    ):
        self.attn_mode   = attn_mode
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.batch_size  = batch_size
        self.random_seed = random_seed
        self.device      = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._fp_selector:    FTv4_5                 = FTv4_5()
        self._chem_encoder:   ChemBERTaEncoder       = ChemBERTaEncoder()
        self._selected_cols:  Optional[List[str]]    = None
        self._encoder:        Optional[_FTv6Encoder] = None

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, smiles: pd.Series) -> None:
        """FTv4.5 FP 선택 → ChemBERTa 1회 캐싱 → 인코더 지도 학습."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Step 1: FP feature selection
        print("[FTv6] Step 1: FTv4.5 FP 피처 선택 중...")
        self._selected_cols = self._fp_selector.select_features(X, y)
        fp_dim = len(self._selected_cols)
        print(f"[FTv6] 선택된 FP 피처: {fp_dim}개 → Linear({fp_dim}, 16)")

        # Step 2: ChemBERTa 1회 전체 인코딩 (캐싱)
        print("[FTv6] Step 2: ChemBERTa 임베딩 사전 계산 (1회)...")
        chem_cache_np = self._chem_encoder.encode_all(
            smiles.tolist(), self.device, batch_size=64
        )  # (n_train, 768)

        # Step 3: 인코더 초기화
        self._encoder = _FTv6Encoder(fp_dim=fp_dim, attn_mode=self.attn_mode).to(self.device)

        # Step 4: 지도 학습 (ChemBERTa 캐시 사용)
        print(f"[FTv6] Step 3: 인코더 학습 ({self.n_epochs} epochs, device={self.device})")
        self._train(X[self._selected_cols], y, chem_cache_np)
        print("[FTv6] 인코더 학습 완료.")

    def transform(self, X: pd.DataFrame, smiles: pd.Series) -> np.ndarray:
        """(n_samples, 16) 피처 행렬 반환."""
        assert self._encoder is not None and self._selected_cols is not None, \
            "[FTv6] transform() 전에 fit()을 실행하세요."

        # ChemBERTa 인코딩 (전체 1회)
        print("[FTv6] ChemBERTa 임베딩 계산 중 (transform)...")
        chem_np = self._chem_encoder.encode_all(
            smiles.tolist(), self.device, batch_size=64
        )

        self._encoder.eval()
        fp_t   = torch.tensor(
            X[self._selected_cols].values.astype(np.float32)
        ).to(self.device)
        chem_t = torch.tensor(chem_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            fused = self._encoder(fp_t, chem_t)

        return fused.cpu().numpy()  # (n_samples, 16)

    # ------------------------------------------------------------------
    # 내부 학습 루프
    # ------------------------------------------------------------------

    def _train(
        self,
        X_fp:         pd.DataFrame,
        y:            pd.Series,
        chem_cache_np: np.ndarray,   # (n_train, 768) pre-computed
    ) -> None:
        head      = nn.Linear(self.HIDDEN_DIM, 1).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(head.parameters()),
            lr=self.lr,
        )

        X_np    = X_fp.values.astype(np.float32)
        y_np    = y.values.astype(np.float32)
        chem_np = chem_cache_np.astype(np.float32)
        n       = len(X_np)

        self._encoder.train()
        head.train()

        for epoch in range(self.n_epochs):
            idx        = np.random.permutation(n)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]

                fp_batch   = torch.tensor(X_np[batch_idx]).to(self.device)
                chem_batch = torch.tensor(chem_np[batch_idx]).to(self.device)
                y_batch    = torch.tensor(y_np[batch_idx]).to(self.device)

                optimizer.zero_grad()
                fused  = self._encoder(fp_batch, chem_batch)
                logits = head(fused).squeeze(-1)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  [FTv6] epoch {epoch + 1:3d}/{self.n_epochs}  loss={epoch_loss:.4f}")
