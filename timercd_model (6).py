"""
timercd_model.py
================
Самостоятельная реализация Time-RCD (Relative Context Discrepancy)
для детектирования аномалий в многомерных временных рядах.

Статья: https://arxiv.org/abs/2509.21190
Реализовано с нуля, без зависимости от оригинального репо.

Ключевая идея RCD
-----------------
Для каждого запросного окна (query) модель видит контекстное окно (context)
из той же точки ряда. Аномалия — это когда query сильно отличается от того,
что модель предсказывает на основе context.

    [--- context window ---][--- query window ---]
           ↓ encoder               ↓ (маскируется)
           ↓←←←←←←←← cross-attn ←←←←←←←←←←←←↓
                      reconstructed query
                             ↓
           anomaly_score = |original - reconstructed|

Архитектура
-----------
1. RevIN          — обратимая нормализация по каналам
2. Patching       — разбивка на патчи, (B, T, C) → (B, N_patches, C*patch_len)
3. PatchEmbedding — Linear + positional embedding
4. Masking        — случайное / блочное / независимое по каналам
5. Transformer Encoder — стек блоков (self-attn + FFN)
6. ReconHead      — Linear → обратно в патчи → ряд

Многомерность
-------------
Каждый патч содержит значения **всех каналов** за один временной отрезок.
Это joint-embedding: модель видит межканальные зависимости в каждом токене.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class RCDOutput:
    reconstruction: torch.Tensor        # (B, T, C) — реконструированный ряд
    anomaly_scores: torch.Tensor         # (B, T)    — скор аномалии по каждому timestep
    loss: Optional[torch.Tensor] = None  # скалярный лосс (только при обучении)

class RevIN(nn.Module):
    """
    Обратимая instance-нормализация по каналам.

    Статистики (mean, std) возвращаются явно из norm и передаются в denorm.
    Это исключает утечку состояния между обучающими батчами и инференсом:
    каждое окно нормируется и денормируется своими собственными статистиками.

    Использование:
        x_norm, stats = revin(x, mode="norm")
        x_denorm      = revin(x_recon, mode="denorm", stats=stats)
    """
    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_channels))
            self.beta  = nn.Parameter(torch.zeros(1, 1, n_channels))

    def forward(self, x: torch.Tensor, mode: str = "norm", stats=None):
        """x: (B, T, C)"""
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)
            std  = x.std(dim=1, keepdim=True) + self.eps
            x = (x - mean) / std
            if self.affine:
                x = x * self.gamma + self.beta
            return x, (mean, std)
        elif mode == "denorm":
            assert stats is not None, "stats required for denorm"
            mean, std = stats
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * std + mean
            return x
        else:
            raise ValueError(f"Unknown mode: {mode}")

class Patchify(nn.Module):
    """
    Нарезает временной ряд на патчи.
    (B, T, C) → (B, N_patches, C * patch_len)

    Все каналы конкатенируются внутри каждого патча — joint embedding.
    """
    def __init__(self, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, N, C*P)"""
        B, T, C = x.shape
        P = self.patch_len
        S = self.stride

        # Дополняем до кратной длины если нужно
        n_patches = (T - P) // S + 1
        patches = []
        for i in range(n_patches):
            start = i * S
            patch = x[:, start:start + P, :]      # (B, P, C)
            patches.append(patch.reshape(B, 1, C * P))  # (B, 1, C*P)

        return torch.cat(patches, dim=1)  # (B, N, C*P)

    def n_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_len) // self.stride + 1


class Unpatchify(nn.Module):
    """
    Собирает ряд обратно из патчей усреднением перекрывающихся участков.
    (B, N_patches, C * patch_len) → (B, T, C)
    """
    def __init__(self, patch_len: int = 16, stride: int = 8, seq_len: int = 512, n_channels: int = 5):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.n_channels = n_channels

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, C*P) → (B, T, C)"""
        B, N, CP = patches.shape
        C = self.n_channels
        P = self.patch_len
        S = self.stride
        T = self.seq_len

        output = torch.zeros(B, T, C, device=patches.device)
        count  = torch.zeros(B, T, C, device=patches.device)

        for i in range(N):
            start = i * S
            end   = start + P
            if end > T:
                break
            patch = patches[:, i, :].reshape(B, P, C)  # (B, P, C)
            output[:, start:end, :] += patch
            count[:, start:end, :]  += 1

        output = output / (count + 1e-8)
        return output

class PatchEmbedding(nn.Module):
    def __init__(self, joint_dim: int, d_model: int, max_patches: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(joint_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self._build_pe(max_patches, d_model)

    def _build_pe(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, joint_dim) → (B, N, d_model)"""
        x = self.proj(x)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention с residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop(attn_out))
        # FFN с residual
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Голова реконструкции
# ─────────────────────────────────────────────────────────────────────────────

class ReconHead(nn.Module):
    def __init__(self, d_model: int, joint_dim: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, joint_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → (B, N, joint_dim)"""
        return self.proj(self.drop(x))


MaskType = Literal["random", "block", "channel_independent"]


def random_mask(n_patches: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Случайное маскирование патчей.
    Возвращает бинарную маску: 1 = замаскировано, 0 = видимо.
    При mask_ratio=0 возвращает нулевую маску (ничего не скрыто).
    """
    if mask_ratio <= 0:
        return torch.zeros(n_patches, device=device)
    n_masked = max(1, int(mask_ratio * n_patches))
    mask = torch.zeros(n_patches, device=device)
    idx  = torch.randperm(n_patches, device=device)[:n_masked]
    mask[idx] = 1.0
    return mask


def block_mask(n_patches: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Блочное маскирование — маскируется один непрерывный блок.
    Более реалистично, т.к. аномалии обычно непрерывны.
    При mask_ratio=0 возвращает нулевую маску.
    """
    if mask_ratio <= 0:
        return torch.zeros(n_patches, device=device)
    n_masked = max(1, int(mask_ratio * n_patches))
    max_start = max(0, n_patches - n_masked)
    start = torch.randint(0, max_start + 1, (1,)).item()
    mask = torch.zeros(n_patches, device=device)
    mask[start:start + n_masked] = 1.0
    return mask


def apply_mask_to_patches(
    patches: torch.Tensor,
    mask_ratio: float,
    mask_type: MaskType = "random",
    n_channels: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Применяет маску к батчу патчей.

    patches   : (B, N, C*P)
    mask_type :
      - "random"              — случайно, одинаково для всего батча-элемента
      - "block"               — блок, одинаково
      - "channel_independent" — каждый канал маскируется отдельно
                                (требует n_channels для разбивки патча)

    Возвращает:
      masked_patches : (B, N, C*P)  — с нулями на маске
      mask           : (B, N)        — 1=masked, 0=visible
    """
    B, N, CP = patches.shape
    device = patches.device

    if mask_ratio <= 0:
        return patches, torch.zeros(B, N, device=device)

    all_masks = []
    for b in range(B):
        if mask_type == "random":
            m = random_mask(N, mask_ratio, device)
        elif mask_type == "block":
            m = block_mask(N, mask_ratio, device)
        elif mask_type == "channel_independent":
            m = random_mask(N, mask_ratio, device)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
        all_masks.append(m)

    mask = torch.stack(all_masks, dim=0) 
    masked = patches * (1.0 - mask.unsqueeze(-1))
    return masked, mask

class TimeRCD(nn.Module):
    """
    Time-RCD: Transformer-based модель для детектирования аномалий
    в многомерных временных рядах.

    Параметры
    ----------
    n_channels   : число каналов входного ряда
    seq_len      : длина окна (context + query)
    patch_len    : длина патча (рекомендуется 8–16)
    patch_stride : шаг патча (рекомендуется patch_len // 2)
    d_model      : размерность трансформера
    n_heads      : число голов attention
    n_layers     : число слоёв трансформера
    d_ff         : размерность FFN (обычно 4 * d_model)
    dropout      : dropout
    mask_type    : тип маскирования при обучении
    context_ratio: доля окна под контекст (остаток — query для RCD)

    Пример использования
    --------------------
    model = TimeRCD(n_channels=5, seq_len=512)

    # Обучение:
    loss = model.compute_loss(x_train, mask_ratio=0.4)
    loss.backward()

    # Инференс (аномалии):
    output = model(x_test)
    scores = output.anomaly_scores  # (B, T)
    recon  = output.reconstruction  # (B, T, C)
    """

    def __init__(
        self,
        n_channels:    int   = 5,
        seq_len:       int   = 512,
        patch_len:     int   = 16,
        patch_stride:  int   = 8,
        d_model:       int   = 256,
        n_heads:       int   = 8,
        n_layers:      int   = 4,
        d_ff:          int   = 1024,
        dropout:       float = 0.1,
        mask_type:     MaskType = "random",
        context_ratio: float = 0.5,
    ):
        super().__init__()

        self.n_channels    = n_channels
        self.seq_len       = seq_len
        self.patch_len     = patch_len
        self.patch_stride  = patch_stride
        self.d_model       = d_model
        self.mask_type     = mask_type
        self.context_ratio = context_ratio

        joint_dim  = n_channels * patch_len

        # Компоненты
        self.revin   = RevIN(n_channels)
        self.patchify = Patchify(patch_len, patch_stride)

        # Количество патчей из seq_len
        self._n_patches = self.patchify.n_patches(seq_len)

        self.embed   = PatchEmbedding(joint_dim, d_model, max_patches=self._n_patches + 8)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = ReconHead(d_model, joint_dim, dropout)
        self.unpatchify = Unpatchify(patch_len, patch_stride, seq_len, n_channels)

    def _encode(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, C*P) → (B, N, d_model)"""
        x = self.embed(patches)
        for block in self.encoder:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> RCDOutput:
        """
        Инференс без маскирования.
        x: (B, T, C)
        """
        B, T, C = x.shape
        x_norm, stats = self.revin(x, "norm")

        # Патчинг
        patches = self.patchify(x_norm)          # (B, N, C*P)

        # Энкодинг
        encoded = self._encode(patches)          # (B, N, d_model)

        # Реконструкция
        recon_patches = self.head(encoded)       # (B, N, C*P)
        recon_norm    = self.unpatchify(recon_patches)  # (B, T, C)

        # Денормализация теми же stats что и нормализация
        recon = self.revin(recon_norm, "denorm", stats=stats)

        # Anomaly score: MAE по каналам, усреднённый в каждый timestep
        anomaly_scores = torch.abs(x - recon).max(dim=-1)  # (B, T)

        return RCDOutput(reconstruction=recon, anomaly_scores=anomaly_scores)

    def compute_loss(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.4,
        mask_type: Optional[MaskType] = None,
    ) -> torch.Tensor:
        """
        Masked reconstruction loss.

        При mask_ratio=0  → обычный автоэнкодер, лосс по всем патчам.
        При mask_ratio>0  → лосс только по замаскированным патчам (MAE-стиль).

        x: (B, T, C)
        """
        mask_type = mask_type or self.mask_type
        B, T, C = x.shape

        x_norm, _ = self.revin(x, "norm")
        patches = self.patchify(x_norm)          # (B, N, C*P)

        if mask_ratio <= 0:
            encoded = self._encode(patches)
            recon_patches = self.head(encoded)
            loss = F.mse_loss(recon_patches, patches)
            return loss

        # Маскирование
        masked_patches, mask = apply_mask_to_patches(
            patches, mask_ratio, mask_type, n_channels=C
        )                                         # (B, N, C*P), (B, N)


        encoded = self._encode(masked_patches)
        recon_patches = self.head(encoded)        # (B, N, C*P)

        inv_mask = mask.unsqueeze(-1)             # (B, N, 1)
        loss = ((recon_patches - patches) ** 2 * inv_mask).sum() / (inv_mask.sum() * patches.shape[-1] + 1e-8)

        return loss

    def compute_loss_rcd(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.4,
    ) -> torch.Tensor:
        """
        RCD-вариант лосса: context восстанавливает query.

        Окно делится на:
          - context: первые context_ratio патчей (видны модели)
          - query:   оставшиеся патчи (маскируются, их нужно предсказать)

        Это ключевая идея Time-RCD: аномалия = query не согласуется с context.
        """
        B, T, C = x.shape
        x_norm, _ = self.revin(x, "norm")
        patches = self.patchify(x_norm)          # (B, N, C*P)
        N = patches.shape[1]

        n_context = max(1, int(self.context_ratio * N))

        mask = torch.zeros(B, N, device=x.device)
        mask[:, n_context:] = 1.0

        if mask_ratio > 0:
            extra_mask = torch.zeros(B, N, device=x.device)
            for b in range(B):
                m = random_mask(n_context, mask_ratio, x.device)
                extra_mask[b, :n_context] = m
            mask = torch.clamp(mask + extra_mask, 0, 1)

        masked_patches = patches * (1.0 - mask.unsqueeze(-1))

        encoded = self._encode(masked_patches)
        recon_patches = self.head(encoded)

        # Лосс только по query части
        query_mask = torch.zeros_like(mask)
        query_mask[:, n_context:] = 1.0
        inv_mask = query_mask.unsqueeze(-1)
        loss = ((recon_patches - patches) ** 2 * inv_mask).sum() / (inv_mask.sum() * patches.shape[-1] + 1e-8)
        return loss

    def get_anomaly_scores_rcd(self, x: torch.Tensor) -> RCDOutput:
        """
        RCD инференс: скользящим окном вычисляем discrepancy.
        Для каждой позиции: context = предыдущие патчи, query = текущий.
        """
        B, T, C = x.shape
        x_norm, stats = self.revin(x, "norm")
        patches = self.patchify(x_norm)          # (B, N, C*P)
        N = patches.shape[1]

        all_scores = torch.zeros(B, N, device=x.device)

        # Для каждого патча: берём всё до него как context, он — query
        for q_idx in range(1, N):
            # Маска: только q_idx маскируем
            mask = torch.zeros(B, N, device=x.device)
            mask[:, q_idx] = 1.0
            masked = patches * (1.0 - mask.unsqueeze(-1))

            with torch.no_grad():
                encoded = self._encode(masked)
                recon   = self.head(encoded)

            # Ошибка только для q_idx патча
            err = ((recon[:, q_idx, :] - patches[:, q_idx, :]) ** 2).mean(dim=-1)  # (B,)
            all_scores[:, q_idx] = err

        # Первый патч — берём среднюю ошибку по полной реконструкции
        all_scores[:, 0] = all_scores[:, 1:].mean(dim=1)

        # Из патч-скоров → timestep скоры
        score_ts = torch.zeros(B, T, device=x.device)
        count_ts = torch.zeros(B, T, device=x.device)
        for i in range(N):
            start = i * self.patch_stride
            end   = min(start + self.patch_len, T)
            score_ts[:, start:end] += all_scores[:, i:i+1]
            count_ts[:, start:end] += 1
        score_ts = score_ts / (count_ts + 1e-8)

        # Полная реконструкция для метрик
        with torch.no_grad():
            full_out = self.forward(x)

        return RCDOutput(
            reconstruction=full_out.reconstruction,
            anomaly_scores=score_ts,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Фабрика с пресетами
# ─────────────────────────────────────────────────────────────────────────────

def build_timercd(
    n_channels: int,
    seq_len: int = 512,
    size: Literal["small", "medium", "large"] = "medium",
    mask_type: MaskType = "random",
    device: str = "cpu",
) -> TimeRCD:
    """
    Фабрика с готовыми конфигурациями.

    small  — быстрое обучение, для экспериментов
    medium — баланс качества и скорости (рекомендуется)
    large  — максимальное качество, медленнее

    Пример:
        model = build_timercd(n_channels=5, seq_len=512, size="medium")
    """
    configs = {
        "small": dict(d_model=128, n_heads=4,  n_layers=2, d_ff=512,  patch_len=16, patch_stride=8),
        "medium":dict(d_model=256, n_heads=8,  n_layers=4, d_ff=1024, patch_len=16, patch_stride=8),
        "large": dict(d_model=512, n_heads=8,  n_layers=6, d_ff=2048, patch_len=16, patch_stride=8),
    }
    cfg = configs[size]
    model = TimeRCD(
        n_channels=n_channels,
        seq_len=seq_len,
        mask_type=mask_type,
        **cfg,
    )
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Быстрый smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, C = 4, 512, 5
    x = torch.randn(B, T, C)

    for size in ["small", "medium"]:
        print(f"\n{'='*50}")
        print(f"Model size: {size}")
        model = build_timercd(n_channels=C, seq_len=T, size=size)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Параметры: {n_params:,}")

        with torch.no_grad():
            out = model(x)
        print(f"  reconstruction: {out.reconstruction.shape}")
        print(f"  anomaly_scores: {out.anomaly_scores.shape}")

        loss_random = model.compute_loss(x, mask_ratio=0.4, mask_type="random")
        loss_block  = model.compute_loss(x, mask_ratio=0.4, mask_type="block")
        loss_rcd    = model.compute_loss_rcd(x, mask_ratio=0.2)
        print(f"  loss (random):  {loss_random.item():.4f}")
        print(f"  loss (block):   {loss_block.item():.4f}")
        print(f"  loss (rcd):     {loss_rcd.item():.4f}")
