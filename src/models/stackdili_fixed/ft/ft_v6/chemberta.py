import numpy as np
import torch
import torch.nn as nn


class ChemBERTaEncoder(nn.Module):
    """SMILES → ChemBERTa [CLS] raw embedding (768-dim).

    proj 레이어 없음. 768-dim raw 임베딩만 반환.
    proj (768 → 16) 는 _FTv6Encoder 내부에서 학습 파라미터로 관리.

    모델: seyonec/ChemBERTa-zinc-base-v1 (hidden_size=768, RoBERTa 기반)
    로드: RobertaForMaskedLM → .roberta 추출 (weight tying 정상 복원)
    """

    CHEMBERTA_DIM = 768

    def __init__(self):
        super().__init__()
        self._tokenizer = None
        self._backbone  = None

    def load_backbone(self) -> None:
        from transformers import RobertaForMaskedLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        full_model = RobertaForMaskedLM.from_pretrained(
            "seyonec/ChemBERTa-zinc-base-v1",
            use_safetensors=True,
        )
        self._backbone = full_model.roberta
        self._backbone.eval()
        print("[ChemBERTa] 로드 완료 (seyonec/ChemBERTa-zinc-base-v1, 768-dim)")

    def encode_all(
        self,
        smiles_list: list,
        device: torch.device,
        batch_size: int = 64,
    ) -> np.ndarray:
        """전체 SMILES를 배치로 인코딩 → (N, 768) numpy 배열.

        학습 루프 밖에서 1회만 호출해 캐싱하는 용도.
        """
        if self._backbone is None:
            self.load_backbone()

        self._backbone = self._backbone.to(device)
        all_embs = []

        print(f"[ChemBERTa] {len(smiles_list)}개 SMILES 인코딩 중 (batch={batch_size})...")
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                cls_emb = self._backbone(**enc).last_hidden_state[:, 0, :]  # (B, 768)
            all_embs.append(cls_emb.cpu().numpy())

        result = np.vstack(all_embs)  # (N, 768)
        print(f"[ChemBERTa] 인코딩 완료 → shape {result.shape}")
        return result

    def forward(self, smiles: list, device: torch.device) -> torch.Tensor:
        """단일 배치 인코딩 (transform 시 사용). (B, 768)"""
        if self._backbone is None:
            self.load_backbone()
        self._backbone = self._backbone.to(device)
        enc = self._tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            cls_emb = self._backbone(**enc).last_hidden_state[:, 0, :]
        return cls_emb
