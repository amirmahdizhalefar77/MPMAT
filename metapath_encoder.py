"""
metapath_encoder.py
===================
Keras functional sub-graph implementing the MLP+Transformer meta-path encoder
described in MPMAT paper, Section II-E-2.

Four-level batch normalisation hierarchy (Fig. 1, Section II-E-2):
  Level 1 – Input BN          : corrects PT/PD/PDT scale differences   (Eq. 10)
  Level 2 – Intra-MLP BN      : prevents internal covariate shift       (Eq. 11)
  Level 3 – Interface BN      : decouples MLP ↔ Transformer gradients   (Eqs. 13-14)
  Level 4 – Hybrid LN-then-BN : per-sublayer coordination in Transformer (Eqs. 15-16)

Architecture flow:
  Input x ∈ R^{3·Nt}
    → Level-1 BN
    → MLP (3Nt → 512 → 256 → 128): each layer: Linear → BN → ELU → Dropout
    → Level-3 Interface BN
    → Reshape (batch, 1, 128)  [single-token for Transformer]
    → 2 × TransformerEncoderLayer with LN-then-BN hybrid
    → Reshape (batch, 128)
    → zm ∈ R^{128}

Public API:
  build_metapath_encoder(metapath_len, ...) → (Input_tensor, zm_tensor)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _elu_activation(x, alpha=1.0):
    """ELU with α=1.0  (Eq. 12, Section II-E-2)."""
    return tf.where(x > 0, x, alpha * (tf.exp(x) - 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Transformer sub-components (Section II-E-2, Eqs. 15-16)
# ──────────────────────────────────────────────────────────────────────────────

class _MetaPathMHSA(keras.layers.Layer):
    """
    Multi-Head Self-Attention for the meta-path Transformer.
    H=4 heads, dk = dmodel/H = 32  (paper Section II-E-2).
    """
    def __init__(self, dmodel=128, num_heads=4, reg=0.001, **kwargs):
        super().__init__(**kwargs)
        assert dmodel % num_heads == 0
        self.dmodel    = dmodel
        self.num_heads = num_heads
        self.dk        = dmodel // num_heads    # = 32

        _r = l2(reg)
        # Fused QKV projection (one Dense per Q, K, V)
        self.WQ = layers.Dense(dmodel, use_bias=False, kernel_regularizer=_r)
        self.WK = layers.Dense(dmodel, use_bias=False, kernel_regularizer=_r)
        self.WV = layers.Dense(dmodel, use_bias=False, kernel_regularizer=_r)
        self.WO = layers.Dense(dmodel, kernel_regularizer=_r)

    def call(self, x, training=False):
        """x: (batch, seq_len, dmodel)"""
        B  = tf.shape(x)[0]
        L  = tf.shape(x)[1]

        def _split(t):
            # (B, L, dmodel) → (B, num_heads, L, dk)
            t = tf.reshape(t, [B, L, self.num_heads, self.dk])
            return tf.transpose(t, [0, 2, 1, 3])

        Q = _split(self.WQ(x))   # (B, H, L, dk)
        K = _split(self.WK(x))
        V = _split(self.WV(x))

        # Scaled dot-product attention
        scale   = tf.math.sqrt(tf.cast(self.dk, tf.float32))
        scores  = tf.matmul(Q, K, transpose_b=True) / scale  # (B, H, L, L)
        weights = tf.nn.softmax(scores, axis=-1)

        context = tf.matmul(weights, V)           # (B, H, L, dk)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [B, L, self.dmodel])

        return self.WO(context)


class _MetaPathFFN(keras.layers.Layer):
    """
    Position-wise Feed-Forward Network.
    dmodel → dff=256 → dmodel, with ELU activation  (Section II-E-2, Eq. 16).
    """
    def __init__(self, dmodel=128, dff=256, reg=0.001, **kwargs):
        super().__init__(**kwargs)
        _r = l2(reg)
        self.dense1 = layers.Dense(dff,    kernel_regularizer=_r)   # expand
        self.dense2 = layers.Dense(dmodel, kernel_regularizer=_r)   # project back

    def call(self, x, training=False):
        h = _elu_activation(self.dense1(x))
        return self.dense2(h)


class _MetaPathEncoderLayer(keras.layers.Layer):
    """
    Single Transformer encoder layer implementing Level-4 hybrid LN-then-BN
    normalisation (Eqs. 15-16):

      z_attn = BN_attn( LN( z^(l-1) + Dropout(MHSA(z^(l-1))) ) )
      z^(l)  = BN_ffn ( LN( z_attn  + Dropout(FFN(z_attn))   ) )
    """
    def __init__(self, dmodel=128, num_heads=4, dff=256,
                 dropout_rate=0.1, reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.mhsa    = _MetaPathMHSA(dmodel, num_heads, reg)
        self.ffn     = _MetaPathFFN(dmodel, dff, reg)

        # Layer Normalization (inside residual stream)
        self.ln_attn = layers.LayerNormalization(epsilon=1e-6)
        self.ln_ffn  = layers.LayerNormalization(epsilon=1e-6)

        # Batch Normalization (at sublayer OUTPUT – Level 4)
        self.bn_attn = layers.BatchNormalization()
        self.bn_ffn  = layers.BatchNormalization()

        self.drop_attn = layers.Dropout(dropout_rate)
        self.drop_ffn  = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # ── Attention sub-layer ──────────────────────────────────────────────
        # LN on the residual INPUT (preserves relative embedding magnitudes)
        attn_out  = self.mhsa(x, training=training)
        attn_out  = self.drop_attn(attn_out, training=training)
        # BN on the sublayer OUTPUT (coordinates mini-batch statistics – Eq. 15)
        z_attn    = self.bn_attn(self.ln_attn(x + attn_out), training=training)

        # ── FFN sub-layer ───────────────────────────────────────────────────
        ffn_out   = self.ffn(z_attn, training=training)
        ffn_out   = self.drop_ffn(ffn_out, training=training)
        z_out     = self.bn_ffn(self.ln_ffn(z_attn + ffn_out), training=training)

        return z_out


# ──────────────────────────────────────────────────────────────────────────────
# Public API: build_metapath_encoder
# ──────────────────────────────────────────────────────────────────────────────

def build_metapath_encoder(
        metapath_len,          # 3 × Nt  (e.g. 3759 for 1253 proteins)
        dmodel=128,            # output dimension  (zm)
        num_heads=4,           # Transformer heads (dk = dmodel/num_heads = 32)
        dff=256,               # Transformer FFN hidden dimension
        num_layers=2,          # number of Transformer encoder layers
        dropout_rate=0.1,      # dropout throughout
        mlp_dims=(512, 256),   # MLP hidden dimensions (before final dmodel)
        reg=0.001,             # L2 regularisation coefficient
):
    """
    Build the MLP+Transformer meta-path encoder sub-graph.

    Returns
    -------
    metapath_input : keras.Input  shape=(None, metapath_len)
    zm             : Tensor       shape=(None, dmodel=128)

    Usage in main.py:
        metapath_input, zm = build_metapath_encoder(metapath_len=3*Nt)
        # then include metapath_input in Model(..., inputs=[..., metapath_input])
        # and zm in Concatenate()([zm, finalmodel_D, finalmodel_P])
    """
    _r = l2(reg)

    # ── Input ────────────────────────────────────────────────────────────────
    metapath_input = keras.Input(shape=(metapath_len,), name='metapath_input')

    # ── Level 1: Input Batch Normalisation (Eq. 10) ──────────────────────────
    # Corrects PT / PD / PDT scale differences before any learnable transform.
    x = layers.BatchNormalization(name='bn_input_level1')(metapath_input)

    # ── Level 2: Intra-MLP Batch Normalisation (Eq. 11) ──────────────────────
    # Dimensionality schedule: 3Nt → 512 → 256 → 128
    # Each layer: Linear → BN → ELU → Dropout
    dims = list(mlp_dims) + [dmodel]      # [512, 256, 128]
    for l_idx, units in enumerate(dims):
        x = layers.Dense(
            units,
            use_bias=False,               # BN has its own bias
            kernel_regularizer=_r,
            name=f'mlp_linear_{l_idx}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_mlp_level2_{l_idx}')(x)
        # ELU after BN (preserves smooth gradient; Eq. 12)
        x = layers.Lambda(
            lambda t: _elu_activation(t, alpha=1.0),
            name=f'mlp_elu_{l_idx}'
        )(x)
        x = layers.Dropout(dropout_rate, name=f'mlp_dropout_{l_idx}')(x)

    # ── Level 3: Interface Batch Normalisation (Eq. 13) ──────────────────────
    # Decouples MLP ↔ Transformer: forward pass standardises Transformer input;
    # backward pass bounds gradient magnitude entering MLP via γ_int (Eq. 14).
    x = layers.BatchNormalization(name='bn_interface_level3')(x)

    # ── Reshape for Transformer (single-token sequence) ──────────────────────
    # x: (batch, 128) → (batch, 1, 128)
    x = layers.Reshape((1, dmodel), name='reshape_for_transformer')(x)

    # ── Level 4: Transformer Encoder (Eqs. 15-16) ────────────────────────────
    # LTR=2 layers, H=4 heads, dk=32, dff=256
    for t_idx in range(num_layers):
        enc_layer = _MetaPathEncoderLayer(
            dmodel=dmodel,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            reg=reg,
            name=f'transformer_enc_{t_idx}'
        )
        x = enc_layer(x)

    # ── Reshape back to (batch, dmodel) ──────────────────────────────────────
    zm = layers.Reshape((dmodel,), name='zm_output')(x)

    return metapath_input, zm


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Nt           = 1253
    metapath_len = 3 * Nt   # 3759

    inp, zm = build_metapath_encoder(metapath_len=metapath_len)
    demo    = keras.Model(inputs=inp, outputs=zm)
    demo.summary()

    x_test = tf.random.normal((4, metapath_len))
    out    = demo(x_test, training=False)
    print("Output shape:", out.shape)   # Expected: (4, 128)
