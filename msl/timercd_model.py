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
4. Masking        — случайное / блочное / независимое по каналам /
                    канальное / гибридное
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
    reconstruction: torch.Tensor         # (B, T, C) — реконструированный ряд
    anomaly_scores: torch.Tensor          # (B, T)    — скор аномалии по каждому timestep
    loss: Optional[torch.Tensor] = None   # скалярный лосс (только при обучении)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RevIN — обратимая нормализация
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Patching / Unpatching
# ─────────────────────────────────────────────────────────────────────────────

class Patchify(nn.Module):
    """
    Нарезает временной ряд на патчи.
    (B, T, C) → (B, N_patches, C * patch_len)

    Все каналы конкатенируются внутри каждого патча — joint embedding.
    """
    def __init__(self, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, N, C*P)"""
        B, T, C = x.shape
        P, S    = self.patch_len, self.stride
        n_patches = (T - P) // S + 1
        patches = []
        for i in range(n_patches):
            start = i * S
            patch = x[:, start:start + P, :]           # (B, P, C)
            patches.append(patch.reshape(B, 1, C * P))  # (B, 1, C*P)
        return torch.cat(patches, dim=1)                # (B, N, C*P)

    def n_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_len) // self.stride + 1


class Unpatchify(nn.Module):
    """
    Собирает ряд обратно из патчей усреднением перекрывающихся участков.
    (B, N_patches, C * patch_len) → (B, T, C)
    """
    def __init__(self, patch_len: int = 16, stride: int = 8,
                 seq_len: int = 512, n_channels: int = 5):
        super().__init__()
        self.patch_len  = patch_len
        self.stride     = stride
        self.seq_len    = seq_len
        self.n_channels = n_channels

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, C*P) → (B, T, C)"""
        B, N, CP = patches.shape
        C, P, S, T = self.n_channels, self.patch_len, self.stride, self.seq_len

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

        return output / (count + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PatchEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, joint_dim: int, d_model: int,
                 max_patches: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(joint_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self._build_pe(max_patches, d_model)

    def _build_pe(self, max_len: int, d_model: int):
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, joint_dim) → (B, N, d_model)"""
        x = self.proj(x)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer Encoder Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop(attn_out))
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. Маскирование
# ─────────────────────────────────────────────────────────────────────────────

MaskType = Literal[
    "random",               # случайные патчи
    "block",                # один непрерывный блок
    "channel_independent",  # каждый канал — своя случайная маска
    "channel_wise",         # маскируются целые каналы
    "hybrid",               # block (ratio/2) + random (ratio/2)
]


def random_mask(n_patches: int, mask_ratio: float,
                device: torch.device) -> torch.Tensor:
    """
    Случайное маскирование патчей.
    Возвращает бинарную маску: 1 = замаскировано, 0 = видимо.
    """
    if mask_ratio <= 0:
        return torch.zeros(n_patches, device=device)
    n_masked = max(1, int(mask_ratio * n_patches))
    mask     = torch.zeros(n_patches, device=device)
    idx      = torch.randperm(n_patches, device=device)[:n_masked]
    mask[idx] = 1.0
    return mask


def block_mask(n_patches: int, mask_ratio: float,
               device: torch.device) -> torch.Tensor:
    """
    Блочное маскирование — один непрерывный блок.
    """
    if mask_ratio <= 0:
        return torch.zeros(n_patches, device=device)
    n_masked  = max(1, int(mask_ratio * n_patches))
    max_start = max(0, n_patches - n_masked)
    start     = torch.randint(0, max_start + 1, (1,)).item()
    mask      = torch.zeros(n_patches, device=device)
    mask[start:start + n_masked] = 1.0
    return mask


def apply_mask_to_patches(
    patches:    torch.Tensor,
    mask_ratio: float,
    mask_type:  MaskType = "random",
    n_channels: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Применяет маску к батчу патчей.

    patches    : (B, N, C*P)
    mask_ratio : доля маскируемых единиц (патчей или каналов)
    mask_type  :
      "random"              — случайные патчи, независимо для каждого элемента батча
      "block"               — один непрерывный блок патчей
      "channel_independent" — каждый канал маскируется своей случайной маской
      "channel_wise"        — маскируются целые каналы (все их патчи обнуляются)
      "hybrid"              — block (ratio/2) + random (ratio/2) поверх

    Возвращает:
      masked_patches : (B, N, C*P)  — с нулями на замаскированных позициях
      mask           : (B, N)        — 1 = замаскировано, 0 = видимо
    """
    B, N, CP = patches.shape
    device   = patches.device

    if mask_ratio <= 0:
        return patches, torch.zeros(B, N, device=device)

    all_masks  = []
    all_masked = []

    for b in range(B):

        if mask_type == "random":
            m      = random_mask(N, mask_ratio, device)
            masked = patches[b] * (1.0 - m.unsqueeze(-1))

        elif mask_type == "block":
            m      = block_mask(N, mask_ratio, device)
            masked = patches[b] * (1.0 - m.unsqueeze(-1))

        elif mask_type == "channel_wise":
            # маскируем целые каналы: все патчи канала обнуляются
            assert n_channels is not None, "n_channels required for channel_wise"
            P           = CP // n_channels
            n_masked_ch = max(1, int(n_channels * mask_ratio))
            masked      = patches[b].clone()                    # (N, C*P)
            ch_order    = torch.randperm(n_channels, device=device)
            masked_chs  = ch_order[:n_masked_ch]
            for c in masked_chs.tolist():
                masked[:, c * P:(c + 1) * P] = 0.0             # обнуляем канал
            # patch-level маска: 1 везде, т.к. каждый патч содержит
            # замаскированные каналы
            m = (torch.ones(N, device=device) if n_masked_ch > 0
                 else torch.zeros(N, device=device))

        elif mask_type == "channel_independent":
            # каждый канал маскируется своей независимой случайной маской
            assert n_channels is not None, "n_channels required for channel_independent"
            P        = CP // n_channels
            masked   = patches[b].clone()                       # (N, C*P)
            ch_masks = []
            for c in range(n_channels):
                cm = (torch.rand(N, device=device) < mask_ratio).float()  # (N,)
                masked[:, c * P:(c + 1) * P] *= (1.0 - cm.unsqueeze(-1))
                ch_masks.append(cm)
            # patch-level маска = 1, если хотя бы один канал замаскирован
            m = torch.stack(ch_masks, dim=1).any(dim=1).float()  # (N,)

        elif mask_type == "hybrid":
            # сначала блок (ratio/2), потом случайные патчи (ratio/2) поверх
            m_block  = block_mask(N, mask_ratio * 0.5, device)
            m_random = random_mask(N, mask_ratio * 0.5, device)
            m        = torch.clamp(m_block + m_random, 0, 1)
            masked   = patches[b] * (1.0 - m.unsqueeze(-1))

        else:
            raise ValueError(
                f"Unknown mask_type: '{mask_type}'. "
                f"Choose from: random, block, channel_wise, "
                f"channel_independent, hybrid"
            )

        all_masks.append(m)
        all_masked.append(masked)

    mask           = torch.stack(all_masks,  dim=0)  # (B, N)
    masked_patches = torch.stack(all_masked, dim=0)  # (B, N, C*P)
    return masked_patches, mask


# ─────────────────────────────────────────────────────────────────────────────
# 7. Основная модель TimeRCD
# ─────────────────────────────────────────────────────────────────────────────

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
    """

    def __init__(
        self,
        n_channels:    int      = 5,
        seq_len:       int      = 512,
        patch_len:     int      = 16,
        patch_stride:  int      = 8,
        d_model:       int      = 256,
        n_heads:       int      = 8,
        n_layers:      int      = 4,
        d_ff:          int      = 1024,
        dropout:       float    = 0.1,
        mask_type:     MaskType = "random",
        context_ratio: float    = 0.5,
    ):
        super().__init__()

        self.n_channels    = n_channels
        self.seq_len       = seq_len
        self.patch_len     = patch_len
        self.patch_stride  = patch_stride
        self.d_model       = d_model
        self.mask_type     = mask_type
        self.context_ratio = context_ratio

        joint_dim       = n_channels * patch_len
        self.mask_token = nn.Parameter(torch.zeros(1, 1, joint_dim))

        self.revin      = RevIN(n_channels)
        self.patchify   = Patchify(patch_len, patch_stride)
        self._n_patches = self.patchify.n_patches(seq_len)

        self.embed   = PatchEmbedding(
            joint_dim, d_model,
            max_patches=self._n_patches + 8
        )
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head       = ReconHead(d_model, joint_dim, dropout)
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
        x_norm, stats = self.revin(x, "norm")
        patches       = self.patchify(x_norm)          # (B, N, C*P)
        encoded       = self._encode(patches)           # (B, N, d_model)
        recon_patches = self.head(encoded)              # (B, N, C*P)
        recon_norm    = self.unpatchify(recon_patches)  # (B, T, C)
        recon         = self.revin(recon_norm, "denorm", stats=stats)

        anomaly_scores = torch.abs(x - recon).mean(dim=-1)  # (B, T)
        return RCDOutput(reconstruction=recon, anomaly_scores=anomaly_scores)

    def compute_loss(
        self,
        x:          torch.Tensor,
        mask_ratio: float              = 0.4,
        mask_type:  Optional[MaskType] = None,
    ) -> torch.Tensor:
        """
        Masked reconstruction loss.

        При mask_ratio=0  → обычный автоэнкодер, лосс по всем патчам.
        При mask_ratio>0  → лосс по замаскированным + 0.1 * лосс по видимым.

        x: (B, T, C)
        """
        mask_type = mask_type or self.mask_type
        B, T, C   = x.shape

        x_norm, _ = self.revin(x, "norm")
        patches   = self.patchify(x_norm)               # (B, N, C*P)

        if mask_ratio <= 0:
            encoded       = self._encode(patches)
            recon_patches = self.head(encoded)
            return F.mse_loss(recon_patches, patches)

        # получаем маску (B, N) и строим masked_patches с mask_token
        _, mask = apply_mask_to_patches(
            patches, mask_ratio, mask_type, n_channels=C
        )

        mask_token     = self.mask_token.expand(B, patches.shape[1], -1)
        masked_patches = (
            patches * (1.0 - mask.unsqueeze(-1))
            + mask_token * mask.unsqueeze(-1)
        )

        encoded       = self._encode(masked_patches)
        recon_patches = self.head(encoded)               # (B, N, C*P)

        inv_mask     = mask.unsqueeze(-1)                # (B, N, 1)
        visible_mask = 1.0 - inv_mask

        n_patch_dim  = patches.shape[-1]
        loss_masked  = (
            ((recon_patches - patches) ** 2 * inv_mask).sum()
            / (inv_mask.sum() * n_patch_dim + 1e-8)
        )
        loss_visible = (
            ((recon_patches - patches) ** 2 * visible_mask).sum()
            / (visible_mask.sum() * n_patch_dim + 1e-8)
        )

        return loss_masked + 0.1 * loss_visible

    def compute_loss_rcd(
        self,
        x:          torch.Tensor,
        mask_ratio: float = 0.4,
    ) -> torch.Tensor:
        """
        RCD-вариант лосса: context восстанавливает query.

        Окно делится на:
          context : первые (1 - mask_ratio) патчей (видны модели)
          query   : оставшиеся патчи (маскируются, их нужно предсказать)
        """
        B, T, C   = x.shape
        x_norm, _ = self.revin(x, "norm")
        patches   = self.patchify(x_norm)               # (B, N, C*P)
        N         = patches.shape[1]

        n_context = max(1, int((1.0 - mask_ratio) * N))
        mask      = torch.zeros(B, N, device=x.device)
        mask[:, n_context:] = 1.0

        if mask_ratio > 0:
            extra_mask = torch.zeros(B, N, device=x.device)
            for b in range(B):
                if self.mask_type == "block":
                    m = block_mask(n_context, mask_ratio, x.device)
                else:
                    m = random_mask(n_context, mask_ratio, x.device)
                extra_mask[b, :n_context] = m
            mask = torch.clamp(mask + extra_mask, 0, 1)

        mask_token     = self.mask_token.expand(B, N, -1)
        masked_patches = (
            patches * (1.0 - mask.unsqueeze(-1))
            + mask_token * mask.unsqueeze(-1)
        )

        encoded       = self._encode(masked_patches)
        recon_patches = self.head(encoded)

        # лосс только по query части
        query_mask = torch.zeros_like(mask)
        query_mask[:, n_context:] = 1.0
        inv_mask   = query_mask.unsqueeze(-1)
        loss       = (
            ((recon_patches - patches) ** 2 * inv_mask).sum()
            / (inv_mask.sum() * patches.shape[-1] + 1e-8)
        )
        return loss

    @torch.no_grad()
    def get_anomaly_scores_rcd(self, x: torch.Tensor) -> RCDOutput:
        """
        RCD инференс с causal attention mask — один forward pass.
        Каждый патч видит только предыдущие, discrepancy = ошибка предсказания.
        """
        B, T, C   = x.shape
        x_norm, stats = self.revin(x, "norm")
        patches   = self.patchify(x_norm)               # (B, N, C*P)
        N         = patches.shape[1]

        # causal mask: патч i не видит патчи i+1..N-1
        causal_attn_mask = torch.triu(
            torch.ones(N, N, device=x.device), diagonal=1
        ).bool()                                        # (N, N)

        # один forward pass с causal mask
        emb = self.embed(patches)                       # (B, N, d_model)
        for block in self.encoder:
            attn_out, _ = block.self_attn(
                emb, emb, emb, attn_mask=causal_attn_mask
            )
            emb = block.norm1(emb + block.drop(attn_out))
            emb = block.norm2(emb + block.drop(block.ff(emb)))

        recon_patches = self.head(emb)                  # (B, N, C*P)

        # ошибка для каждого патча
        patch_errors = ((recon_patches - patches) ** 2).mean(dim=-1)  # (B, N)

        # патч-скоры → timestep скоры
        score_ts = torch.zeros(B, T, device=x.device)
        count_ts = torch.zeros(B, T, device=x.device)
        for i in range(N):
            start = i * self.patch_stride
            end   = min(start + self.patch_len, T)
            score_ts[:, start:end] += patch_errors[:, i:i+1]
            count_ts[:, start:end] += 1
        score_ts = score_ts / (count_ts + 1e-8)

        # полная реконструкция для RMSE/MAE
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
    seq_len:    int                                    = 512,
    size:       Literal["small", "medium", "large"]   = "medium",
    mask_type:  MaskType                               = "random",
    device:     str                                    = "cpu",
) -> TimeRCD:
    """
    Фабрика с готовыми конфигурациями.

    small  — быстрое обучение, для экспериментов
    medium — баланс качества и скорости (рекомендуется)
    large  — максимальное качество, медленнее
    """
    configs = {
        "small":  dict(d_model=128, n_heads=4, n_layers=2,
                       d_ff=512,  patch_len=16, patch_stride=8),
        "medium": dict(d_model=256, n_heads=8, n_layers=4,
                       d_ff=1024, patch_len=16, patch_stride=8),
        "large":  dict(d_model=512, n_heads=8, n_layers=6,
                       d_ff=2048, patch_len=16, patch_stride=8),
    }
    model = TimeRCD(
        n_channels=n_channels,
        seq_len=seq_len,
        mask_type=mask_type,
        **configs[size],
    )
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, C = 4, 512, 5
    x = torch.randn(B, T, C)

    for size in ["small", "medium"]:
        print(f"\n{'='*60}")
        print(f"Model size: {size}")
        model   = build_timercd(n_channels=C, seq_len=T, size=size)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Параметры: {n_params:,}")

        with torch.no_grad():
            out = model(x)
        print(f"  reconstruction : {out.reconstruction.shape}")
        print(f"  anomaly_scores : {out.anomaly_scores.shape}")

        for mt in ["random", "block", "channel_wise",
                   "channel_independent", "hybrid"]:
            loss = model.compute_loss(x, mask_ratio=0.4, mask_type=mt)
            print(f"  loss ({mt:<22}): {loss.item():.4f}")

        loss_rcd = model.compute_loss_rcd(x, mask_ratio=0.2)
        print(f"  loss (rcd)                    : {loss_rcd.item():.4f}")
