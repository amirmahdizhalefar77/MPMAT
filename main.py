import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Activation,
    Dropout,
    Embedding,
    SpatialDropout1D,
    Concatenate,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    Add,
    MultiHeadAttention,
    LayerNormalization,
    Reshape,
    Lambda,
)

from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
    TerminateOnNaN,
)

from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Import your existing modules
from pardata_me import parse_data
from transformer import Transformer


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class ModelConfig:
    """
    Centralized configuration for all model hyperparameters.
    """

    def __init__(self):
        # Data dimensions
        self.drug_len = 2048
        self.drug_len2 = 100
        self.prot_len = 800
        self.my_matrix_len = 21411

        # Vocabulary sizes (used by protein/drug sequence models)
        self.protein_vocab_size = 474
        self.drug_vocab_size = 42

        # NOTE:
        # meta-path features in pardata_me.py are loaded as a numeric matrix via pd.read_csv(...).values,
        # so we treat them as continuous IID features in this file.
        # The previous Embedding(vocab) approach for meta-path is removed.

        # Architecture
        self.drug_layers = [128, 64]
        self.protein_layers = [64]

        # Meta-path MLP branch sizes (IID)
        self.my_matrix_layers = [512, 256]   # stronger than [64] to avoid underfitting vs Transformer

        # Meta-path fusion / output size
        self.my_matrix_out_dim = 64

        self.fc_layers = [64, 32]
        self.activation = 'relu'

        # Regularization
        self.dropout = 0.3
        self.spatial_dropout = 0.2
        self.l2_weight = 0.001
        self.l2_activity = 0.001

        # Training
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.n_epochs = 100
        self.gradient_clipnorm = 1.0

        # Early stopping
        self.early_stopping_patience = 70
        self.early_stopping_monitor = 'val_auc_roc'

        # Learning rate reduction
        self.lr_reduction_factor = 0.5
        self.lr_reduction_patience = 5
        self.lr_min = 1e-7

        # Protein Transformer settings (existing custom Transformer, token-based)
        self.protein_transformer = {
            'num_layers': 2,
            'model_size': 20,
            'num_heads': 5,
            'dff_size': 64,
            'maxlen': 800,
        }

        # Meta-path Transformer branch settings (tokenized continuous features)
        # maxlen here is used as the number of learned latent tokens (T)
        self.my_matrix_transformer = {
            'num_layers': 2,
            'model_size': 64,   # token width (d)
            'num_heads': 8,
            'dff_size': 128,
            'maxlen': 128,      # number of tokens (T); controls compute/memory
        }

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(config, key, value)
        print(f"Configuration loaded from {filepath}")
        return config

    def print_summary(self):
        print("\n" + "=" * 70)
        print("MODEL CONFIGURATION")
        print("=" * 70)
        print("Architecture:")
        print(f"  Drug layers:          {self.drug_layers}")
        print(f"  Protein layers:       {self.protein_layers}")
        print(f"  MetaPath MLP layers:  {self.my_matrix_layers}")
        print(f"  MetaPath out dim:     {self.my_matrix_out_dim}")
        print(f"  FC layers:            {self.fc_layers}")
        print("\nRegularization:")
        print(f"  Dropout:              {self.dropout}")
        print(f"  L2 weight:            {self.l2_weight}")
        print(f"  Early stopping:       patience={self.early_stopping_patience}")
        print(f"  LR reduction:         factor={self.lr_reduction_factor}, patience={self.lr_reduction_patience}")
        print("\nTraining:")
        print(f"  Learning rate:        {self.learning_rate}")
        print(f"  Batch size:           {self.batch_size}")
        print(f"  Epochs:               {self.n_epochs}")
        print(f"  Gradient clipping:    {self.gradient_clipnorm}")
        print("=" * 70 + "\n")


# ============================================================================
# META-PATH PROCESSOR (BN-MLP + BN-TRANSFORMER + GATED FUSION)
# ============================================================================

class MetaPathTransformerEncoderBlock(keras.layers.Layer):
    """
    Transformer encoder block with:
      - Pre-LayerNorm (standard for Transformers)
      - Residual connections
      - BatchNorm after residual updates (per-mini-batch stabilization)

    BatchNorm axis=-1 normalizes channel dimension using statistics pooled over (batch, tokens).
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        dff: int,
        dropout_rate: float,
        l2_weight: float,
        name: str,
    ):
        super().__init__(name=name)
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.l2_weight = l2_weight

        key_dim = max(1, token_dim // max(1, num_heads))

        self.ln1 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            output_shape=token_dim,
            name=f"{name}_mha",
        )
        self.drop_attn = Dropout(dropout_rate, name=f"{name}_drop_attn")
        self.bn_attn = BatchNormalization(axis=-1, name=f"{name}_bn_attn")

        self.ln2 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")
        self.ffn1 = Dense(
            dff,
            activation='relu',
            kernel_initializer='glorot_normal',
            kernel_regularizer=l2(l2_weight),
            name=f"{name}_ffn1",
        )
        self.ffn2 = Dense(
            token_dim,
            kernel_initializer='glorot_normal',
            kernel_regularizer=l2(l2_weight),
            name=f"{name}_ffn2",
        )
        self.drop_ffn = Dropout(dropout_rate, name=f"{name}_drop_ffn")
        self.bn_ffn = BatchNormalization(axis=-1, name=f"{name}_bn_ffn")
        
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "token_dim":    self.token_dim,
            "num_heads":    self.num_heads,
            "dff":          self.dff,
            "dropout_rate": self.dropout_rate,
            "l2_weight":    self.l2_weight,
        })
        return config

    def call(self, x, training=False):
        # Attention sublayer (PreNorm)
        h = self.ln1(x)
        attn = self.mha(query=h, value=h, key=h, training=training)
        attn = self.drop_attn(attn, training=training)
        x = Add(name=f"{self.name}_res_attn")([x, attn])
        x = self.bn_attn(x, training=training)

        # FFN sublayer (PreNorm)
        h2 = self.ln2(x)
        f = self.ffn1(h2)
        f = self.ffn2(f)
        f = self.drop_ffn(f, training=training)
        x = Add(name=f"{self.name}_res_ffn")([x, f])
        x = self.bn_ffn(x, training=training)
        return x
    



def build_metapath_processor(
    mp_input: tf.Tensor,
    cfg: ModelConfig,
    l2_weight: float,
    dropout_rate: float,
    activation: str,
) -> tf.Tensor:
    """
    Continuous meta-path processor with:
      - Shared BN on input
      - MLP branch: Dense -> BN -> Act -> Dropout (stack)
      - Transformer branch: tokenization -> positional embedding -> (EncoderBlocks with BN) -> pooling
      - Gated fusion: prevents one branch from dominating optimization
    """
    params_dict = {
        'kernel_initializer': 'glorot_normal',
        'kernel_regularizer': l2(l2_weight),
    }

    # Ensure float32
    x0 = Lambda(lambda t: tf.cast(t, tf.float32), name="mp_cast_float")(mp_input)

    # Shared per-mini-batch stabilization
    x0 = BatchNormalization(name="mp_input_bn")(x0)

    # ----------------------------
    # (A) MLP branch (IID)
    # ----------------------------
    m = x0
    for i, units in enumerate(cfg.my_matrix_layers):
        m = Dense(units, **params_dict, name=f"mp_mlp_dense_{i}")(m)
        m = BatchNormalization(name=f"mp_mlp_bn_{i}")(m)
        m = Activation(activation, name=f"mp_mlp_act_{i}")(m)
        m = Dropout(dropout_rate, name=f"mp_mlp_drop_{i}")(m)

    m = Dense(cfg.my_matrix_out_dim, **params_dict, name="mp_mlp_proj")(m)
    m = BatchNormalization(name="mp_mlp_proj_bn")(m)
    m = Activation(activation, name="mp_mlp_proj_act")(m)

    # ----------------------------
    # (B) Transformer branch (tokenized continuous features)
    # ----------------------------
    t_tokens = cfg.my_matrix_transformer['maxlen']         # number of tokens (T)
    t_dim = cfg.my_matrix_transformer['model_size']        # token width (d)
    t_layers = cfg.my_matrix_transformer['num_layers']
    t_heads = cfg.my_matrix_transformer['num_heads']
    t_dff = cfg.my_matrix_transformer['dff_size']

    # Learnable tokenization: (B, D) -> (B, T*d) -> (B, T, d)
    
    # AFTER: two-stage bottleneck projection (~1.5M params)
    t = Dense(512, activation='relu', **params_dict, name="mp_tok_bottleneck")(x0)  # 21411->512
    t = Dense(t_tokens * t_dim, **params_dict, name="mp_tok_proj")(t)               # 512->8192
    
    
    t = Reshape((t_tokens, t_dim), name="mp_tok_reshape")(t)
    t = BatchNormalization(axis=-1, name="mp_tok_bn")(t)
    t = Dropout(dropout_rate, name="mp_tok_drop")(t)

    # Trainable positional embedding (broadcast over batch)





    # AFTER (fixed):
    pos_emb_layer = Embedding(
        input_dim=t_tokens,
        output_dim=t_dim,
        embeddings_initializer='glorot_normal',
        embeddings_regularizer=l2(l2_weight),
        name="mp_pos_embedding",
    )
    # Create batch-aware position indices: (None,) -> (None, T) -> embed -> (None, T, d)
    pos_indices = Lambda(
        lambda x: tf.tile(
            tf.expand_dims(tf.range(t_tokens), axis=0),  # (1, T)
            [tf.shape(x)[0], 1]                           # (B, T)
        ),
        name="mp_pos_indices",
    )(t)
    pos_emb = pos_emb_layer(pos_indices)   # (B, T, d) — correct!
    t = Add(name="mp_add_pos")([t, pos_emb])






    for i in range(t_layers):
        t = MetaPathTransformerEncoderBlock(
            token_dim=t_dim,
            num_heads=t_heads,
            dff=t_dff,
            dropout_rate=dropout_rate,
            l2_weight=l2_weight,
            name=f"mp_encblock_{i}",
        )(t)

    # Pool tokens -> vector
    t = GlobalAveragePooling1D(name="mp_tr_pool")(t)
    t = Dense(cfg.my_matrix_out_dim, **params_dict, name="mp_tr_proj")(t)
    t = BatchNormalization(name="mp_tr_proj_bn")(t)
    t = Activation(activation, name="mp_tr_proj_act")(t)

    # ----------------------------
    # (C) Gated fusion (balanced optimization)
    # ----------------------------
    gate_in = Concatenate(name="mp_gate_concat")([m, t])
    gate = Dense(
        cfg.my_matrix_out_dim,
        activation='sigmoid',
        kernel_initializer='glorot_normal',
        kernel_regularizer=l2(l2_weight),
        name="mp_gate",
    )(gate_in)

    one_minus_gate = Lambda(lambda g: 1.0 - g, name="mp_one_minus_gate")(gate)
    m_g = Lambda(lambda z: z[0] * z[1], name="mp_gate_m")([gate, m])
    t_g = Lambda(lambda z: z[0] * z[1], name="mp_gate_t")([one_minus_gate, t])
    mp = Add(name="mp_fused")([m_g, t_g])

    mp = BatchNormalization(name="mp_fused_bn")(mp)
    mp = Dropout(dropout_rate, name="mp_fused_drop")(mp)

    return mp


# ============================================================================
# MAIN MODEL CLASS
# ============================================================================

class EnhancedDTIModel:
    """
    Enhanced Drug-Target Interaction Prediction Model with improved meta-path processing.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        self.callbacks = []

        self._build_model()
        self._compile_model()
        self._setup_callbacks()

    @staticmethod
    def _return_tuple(value):
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, list):
            return value
        return list(value)

    def _build_model(self):
        cfg = self.config
        params_dict = {
            'kernel_initializer': 'glorot_normal',
            'kernel_regularizer': l2(cfg.l2_weight),
        }

        print("\nBuilding Enhanced DTI Model...")
        print("=" * 70)

        # ====================================================================
        # META-PATH FEATURES (continuous IID vector) - BN-MLP + BN-Transformer
        # ====================================================================
        print("1. Building metapath processor (BN-MLP + BN-Transformer + gated fusion)...")

        my_matrix_inputs = Input(
            shape=(cfg.my_matrix_len,),
            dtype=tf.float32,
            name='my_matrix_input',
        )

        my_matrix_final = build_metapath_processor(
            mp_input=my_matrix_inputs,
            cfg=cfg,
            l2_weight=cfg.l2_weight,
            dropout_rate=cfg.dropout,
            activation=cfg.activation,
        )

        # ====================================================================
        # PROTEIN FEATURES - existing token Transformer (sequential)
        # ====================================================================
        print("2. Building protein feature processor (sequential)...")

        protein_enc_inputs = Input(shape=(cfg.prot_len,), name='protein_encoder_input')
        protein_dec_inputs = Input(shape=(cfg.prot_len,), name='protein_decoder_input')

        protein_transformer = Transformer(
            num_layers=cfg.protein_transformer['num_layers'],
            model_size=cfg.protein_transformer['model_size'],
            num_heads=cfg.protein_transformer['num_heads'],
            dff_size=cfg.protein_transformer['dff_size'],
            vocab_size=cfg.protein_vocab_size + 1,
            maxlen=cfg.protein_transformer['maxlen'],
        )

        protein_output = protein_transformer([protein_enc_inputs, protein_dec_inputs])
        protein_output = SpatialDropout1D(cfg.spatial_dropout)(protein_output)
        protein_output = GlobalMaxPooling1D(name='protein_pooling')(protein_output)
        protein_output = Dense(64, activation='relu', **params_dict, name='protein_dense')(protein_output)
        protein_output = BatchNormalization(name='protein_bn')(protein_output)
        protein_output = Dropout(cfg.dropout, name='protein_dropout')(protein_output)

        # ====================================================================
        # DRUG FEATURES
        # ====================================================================
        print("3. Building drug feature processor...")

        # Morgan fingerprints
        drug_input = Input(shape=(cfg.drug_len,), name='drug_morgan_input')
        drug_model = drug_input

        drug_layer_sizes = self._return_tuple(cfg.drug_layers)
        for i, layer_size in enumerate(drug_layer_sizes):
            drug_model = Dense(layer_size, **params_dict, name=f'drug_dense_{i}')(drug_model)
            drug_model = BatchNormalization(name=f'drug_bn_{i}')(drug_model)
            drug_model = Activation(cfg.activation, name=f'drug_act_{i}')(drug_model)
            drug_model = Dropout(cfg.dropout, name=f'drug_dropout_{i}')(drug_model)

        # Drug sequence
        drug_input2 = Input(shape=(cfg.drug_len2,), name='drug_sequence_input')
        drug_model2 = Embedding(
            cfg.drug_vocab_size, 10,
            embeddings_initializer='glorot_normal',
            embeddings_regularizer=l2(cfg.l2_weight),
            name='drug_seq_embedding',
        )(drug_input2)
        drug_model2 = SpatialDropout1D(cfg.spatial_dropout)(drug_model2)
        drug_model2 = GlobalMaxPooling1D(name='drug_seq_pooling')(drug_model2)

        protein_layer_sizes = self._return_tuple(cfg.protein_layers)
        for i, _layer_size in enumerate(protein_layer_sizes):
            drug_model2 = Dense(64, **params_dict, name=f'drug_seq_dense_{i}')(drug_model2)
            drug_model2 = BatchNormalization(name=f'drug_seq_bn_{i}')(drug_model2)
            drug_model2 = Activation(cfg.activation, name=f'drug_seq_act_{i}')(drug_model2)
            drug_model2 = Dropout(cfg.dropout, name=f'drug_seq_dropout_{i}')(drug_model2)

        # Additional protein embedding+pool features
        protein_conv_input = Input(shape=(cfg.prot_len,), name='protein_conv_input')
        protein_conv = Embedding(
            cfg.protein_vocab_size + 1, 20,
            embeddings_initializer='glorot_normal',
            embeddings_regularizer=l2(cfg.l2_weight),
            name='protein_conv_embedding',
        )(protein_conv_input)
        protein_conv = SpatialDropout1D(cfg.spatial_dropout)(protein_conv)
        protein_conv = GlobalMaxPooling1D(name='protein_conv_pooling')(protein_conv)

        for i, _layer_size in enumerate(protein_layer_sizes):
            protein_conv = Dense(64, **params_dict, name=f'protein_conv_dense_{i}')(protein_conv)
            protein_conv = BatchNormalization(name=f'protein_conv_bn_{i}')(protein_conv)
            protein_conv = Activation(cfg.activation, name=f'protein_conv_act_{i}')(protein_conv)
            protein_conv = Dropout(cfg.dropout, name=f'protein_conv_dropout_{i}')(protein_conv)

        # ====================================================================
        # FEATURE FUSION
        # ====================================================================
        print("4. Fusing all features...")

        final_drug = Concatenate(axis=1, name='drug_concat')([drug_model, drug_model2])
        final_drug = Dense(64, **params_dict, name='drug_fusion')(final_drug)
        final_drug = BatchNormalization(name='drug_fusion_bn')(final_drug)
        final_drug = Dropout(cfg.dropout, name='drug_fusion_dropout')(final_drug)

        final_protein = Concatenate(axis=1, name='protein_concat')(
            [protein_conv, protein_output, my_matrix_final]
        )
        final_protein = Dense(64, **params_dict, name='protein_fusion')(final_protein)
        final_protein = BatchNormalization(name='protein_fusion_bn')(final_protein)
        final_protein = Dropout(cfg.dropout, name='protein_fusion_dropout')(final_protein)

        combined = Concatenate(axis=1, name='final_concat')([final_drug, final_protein])

        fc_layer_sizes = self._return_tuple(cfg.fc_layers)
        for i, fc_size in enumerate(fc_layer_sizes):
            combined = Dense(units=fc_size, **params_dict, name=f'fc_dense_{i}')(combined)
            combined = BatchNormalization(name=f'fc_bn_{i}')(combined)
            combined = Activation(cfg.activation, name=f'fc_act_{i}')(combined)
            combined = Dropout(cfg.dropout, name=f'fc_dropout_{i}')(combined)

        output = Dense(
            1,
            activation='sigmoid',
            activity_regularizer=l2(cfg.l2_activity),
            **params_dict,
            name='output',
        )(combined)

        self.model = Model(
            inputs=[
                drug_input,
                drug_input2,
                protein_conv_input,
                protein_enc_inputs,
                protein_dec_inputs,
                my_matrix_inputs,
            ],
            outputs=output,
            name='Enhanced_DTI_Model',
        )

        print("✓ Model architecture built successfully")
        print("=" * 70 + "\n")

    def _compile_model(self):
        optimizer = Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.gradient_clipnorm,
        )

        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                metrics.AUC(curve='ROC', name='auc_roc'),
                metrics.AUC(curve='PR', name='auc_pr'),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall'),
            ],
        )
        print("✓ Model compiled with enhanced metrics")

    def _setup_callbacks(self):
        log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_enhanced_model"

        self.callbacks = [
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,   # disabled — too expensive with 175M weight layer
                write_graph=False,  # graph export also expensive; disable unless needed
                write_images=False,
                update_freq='epoch',
                profile_batch=0,    # disable profiling too
            ),
            ModelCheckpoint(
                filepath='./best_model_enhanced.weights.h5',
                monitor=self.config.early_stopping_monitor,
                mode='max',
                save_best_only=True,
                save_weights_only=True,   # <-- skips get_config() requirement
                verbose=1,
            ),
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                mode='max',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.lr_reduction_factor,
                patience=self.config.lr_reduction_patience,
                min_lr=self.config.lr_min,
                verbose=1,
            ),
            CSVLogger(filename='training_history_enhanced.csv'),
            TerminateOnNaN(),
        ]

        print("✓ Callbacks configured (TensorBoard, Checkpointing, Early Stopping, LR Reduction, NaN guard)")

    def summary(self):
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("=" * 70 + "\n")
        self.model.summary()
        print("\n" + "=" * 70 + "\n")

    def train(self, train_data: Dict, val_data: Dict, verbose: int = 1):
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Training samples: {len(train_data['Label'])}")
        print(f"Validation samples: {len(val_data['Label'])}")
        print(f"Epochs: {self.config.n_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print("=" * 70 + "\n")

        X_train = [
            train_data['drug_feature'],
            train_data['drug_feature2'],
            train_data['protein_feature'],
            train_data['protein_feature2'],
            train_data['protein_feature2'],
            train_data['my_matrix_feature'],
        ]
        y_train = train_data['Label']

        X_val = [
            val_data['drug_feature'],
            val_data['drug_feature2'],
            val_data['protein_feature'],
            val_data['protein_feature2'],
            val_data['protein_feature2'],
            val_data['my_matrix_feature'],
        ]
        y_val = val_data['Label']

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            callbacks=self.callbacks,
            verbose=verbose,
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70 + "\n")
        return self.history

    def evaluate(self, test_data: Dict, threshold: Optional[float] = None) -> Dict[str, float]:
        print("\n" + "=" * 70)
        print("EVALUATING ON TEST SET")
        print("=" * 70 + "\n")

        X_test = [
            test_data['drug_feature'],
            test_data['drug_feature2'],
            test_data['protein_feature'],
            test_data['protein_feature2'],
            test_data['protein_feature2'],
            test_data['my_matrix_feature'],
        ]
        y_test = test_data['Label']

        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()

        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        precision, recall, _thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        if threshold is None:
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = thresholds_roc[optimal_idx]

        y_pred = (y_pred_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics_dict = {
            'threshold': float(threshold),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'f1_score': float(f1_score(y_test, y_pred)),
            'auc_roc': float(roc_auc),
            'auc_pr': float(pr_auc),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }

        print(f"Optimal Threshold:     {metrics_dict['threshold']:.4f}")
        print(f"Accuracy:              {metrics_dict['accuracy']:.4f}")
        print(f"AUC-ROC:               {metrics_dict['auc_roc']:.4f}")
        print(f"AUC-PR:                {metrics_dict['auc_pr']:.4f}")
        print(f"Sensitivity (Recall):  {metrics_dict['sensitivity']:.4f}")
        print(f"Specificity:           {metrics_dict['specificity']:.4f}")
        print(f"Precision:             {metrics_dict['precision']:.4f}")
        print(f"F1-Score:              {metrics_dict['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN: {tn:6d}  |  FP: {fp:6d}")
        print(f"  FN: {fn:6d}  |  TP: {tp:6d}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        roc_df.to_csv('roc_curve_enhanced.csv', index=False)

        pr_df = pd.DataFrame({'recall': recall, 'precision': precision})
        pr_df.to_csv('pr_curve_enhanced.csv', index=False)

        print("\n✓ ROC and PR curves saved to CSV files")
        print("=" * 70 + "\n")

        return metrics_dict

    def save(self, filepath: str):
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")


# ============================================================================
# VISUALIZATION AND EVALUATION UTILITIES
# ============================================================================

class ModelVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_training_history(self, history_csv: str, save_path: Optional[str] = None):
        history = pd.read_csv(history_csv)

        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('Training History - Enhanced Model', fontsize=16, fontweight='bold')

        metrics_to_plot = [
            ('loss', 'Loss'),
            ('accuracy', 'Accuracy'),
            ('auc_roc', 'AUC-ROC'),
            ('auc_pr', 'AUC-PR'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
        ]

        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]

            if metric in history.columns:
                ax.plot(history['epoch'], history[metric],
                        label=f'Training {title}', linewidth=2)

            val_metric = f'val_{metric}'
            if val_metric in history.columns:
                ax.plot(history['epoch'], history[val_metric],
                        label=f'Validation {title}', linewidth=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, save_path: Optional[str] = None):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")

        plt.show()


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced DTI Prediction Model with BN-stabilized MetaPath MLP+Transformer'
    )

    # Data paths
    parser.add_argument('--dti_dir', default='train.csv', help='Training DTI file')
    parser.add_argument('--drug_dir', default='morgan_train.csv', help='Training drug features')
    parser.add_argument('--protein_dir', default='protein_train.csv', help='Training protein features')
    parser.add_argument('--my_matrix_dir', default='my_matrix_train.csv', help='Training metapath features')

    parser.add_argument('--test-dti-dir', default='valid.csv', help='Validation DTI file')
    parser.add_argument('--test-drug-dir', default='morgan_valid.csv', help='Validation drug features')
    parser.add_argument('--test-protein-dir', default='protein_valid.csv', help='Validation protein features')
    parser.add_argument('--test_my_matrix_dir', default='my_matrix_valid.csv', help='Validation metapath features')

    parser.add_argument('--final-test-dti', default='test.csv', help='Test DTI file')
    parser.add_argument('--final-test-drug', default='morgan_test.csv', help='Test drug features')
    parser.add_argument('--final-test-protein', default='protein_test.csv', help='Test protein features')
    parser.add_argument('--final-test-matrix', default='my_matrix_test.csv', help='Test metapath features')

    # Model hyperparameters
    parser.add_argument('--drug-layers', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--protein-layers', type=int, nargs='+', default=[64])
    parser.add_argument('--my_matrix_layers', type=int, nargs='+', default=[512, 256])
    parser.add_argument('--fc-layers', type=int, nargs='+', default=[64, 32])

    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--n-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Configuration
    parser.add_argument('--config', type=str, help='Load configuration from JSON file')
    parser.add_argument('--save-config', type=str, help='Save configuration to JSON file')

    args = parser.parse_args(args=[])

    # Setup GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"✓ GPU devices available: {len(physical_devices)}")
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
    else:
        print("⚠ No GPU found, using CPU")

    # Create or load configuration
    if args.config:
        config = ModelConfig.load(args.config)
    else:
        config = ModelConfig()
        config.drug_layers = args.drug_layers
        config.protein_layers = args.protein_layers
        config.my_matrix_layers = args.my_matrix_layers
        config.fc_layers = args.fc_layers
        config.learning_rate = args.learning_rate
        config.n_epochs = args.n_epoch
        config.batch_size = args.batch_size
        config.dropout = args.dropout

    if args.save_config:
        config.save(args.save_config)

    config.print_summary()

    # Type parameters for data loading
    type_params = {
        'prot_vec': 'Convolution',
        'my_matrix_vec': 'Dense',
        'prot_len': config.prot_len,
        'drug_vec': 'morgan_fp',
        'drug_len': config.drug_len,
        'drug_len2': config.drug_len2,
        'my_matrix_len': config.my_matrix_len,
    }

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    print("Loading training data...")
    train_data = parse_data(
        args.dti_dir,
        args.drug_dir,
        args.protein_dir,
        args.my_matrix_dir,
        **type_params
    )
    print(f"✓ Training samples: {len(train_data['Label'])}")

    print("Loading validation data...")
    val_data = parse_data(
        args.test_dti_dir,
        args.test_drug_dir,
        args.test_protein_dir,
        args.test_my_matrix_dir,
        **type_params
    )
    print(f"✓ Validation samples: {len(val_data['Label'])}")

    print("Loading test data...")
    test_data = parse_data(
        args.final_test_dti,
        args.final_test_drug,
        args.final_test_protein,
        args.final_test_matrix,
        **type_params
    )
    print(f"✓ Test samples: {len(test_data['Label'])}")
    print("=" * 70 + "\n")

    model = EnhancedDTIModel(config)
    model.summary()
    
    print("=== SHAPE DIAGNOSTICS ===")
    for name, data in [("train", train_data), ("val", val_data)]:
        print(f"\n{name}:")
    for key, val in data.items():
        print(f"  {key}: {val.shape}")
    print("=========================")
    
    print("my_matrix_feature shape:", train_data["my_matrix_feature"].shape)
    # Should be (N_samples, 21411)
    # If it's (N_samples, something_else), you have a column count mismatch
    
    # After loading, before building the model:
    actual_matrix_len = train_data["my_matrix_feature"].shape[1]
    print(f"Actual my_matrix_len: {actual_matrix_len}")  # e.g. might be 100, not 21411
    config.my_matrix_len = actual_matrix_len  # <-- sync config to real data
    
    n = len(train_data["Label"])
    for key, val in train_data.items():
        assert val.shape[0] == n, f"Row mismatch in {key}: {val.shape[0]} vs {n}"
    print(f"All features aligned: {n} samples ✓")

    model.train(train_data, val_data)

    metrics_out = model.evaluate(test_data)

    model.save('final_enhanced_model.h5')

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70 + "\n")

    visualizer = ModelVisualizer()

    if os.path.exists('training_history_enhanced.csv'):
        visualizer.plot_training_history(
            'training_history_enhanced.csv',
            save_path='training_history_plot.png',
        )

    X_test = [
        test_data['drug_feature'],
        test_data['drug_feature2'],
        test_data['protein_feature'],
        test_data['protein_feature2'],
        test_data['protein_feature2'],
        test_data['my_matrix_feature'],
    ]
    y_test = test_data['Label']
    y_pred_proba = model.model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= metrics_out['threshold']).astype(int)

    visualizer.plot_confusion_matrix(
        y_test,
        y_pred,
        save_path='confusion_matrix.png',
    )

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - best_model_enhanced.h5          (best checkpoint)")
    print("  - final_enhanced_model.h5         (final model)")
    print("  - training_history_enhanced.csv   (training logs)")
    print("  - roc_curve_enhanced.csv          (ROC data)")
    print("  - pr_curve_enhanced.csv           (PR data)")
    print("  - training_history_plot.png       (training plots)")
    print("  - confusion_matrix.png            (confusion matrix)")
    print("\nFinal Metrics:")
    print(f"  Accuracy:   {metrics_out['accuracy']:.4f}")
    print(f"  AUC-ROC:    {metrics_out['auc_roc']:.4f}")
    print(f"  AUC-PR:     {metrics_out['auc_pr']:.4f}")
    print(f"  F1-Score:   {metrics_out['f1_score']:.4f}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()