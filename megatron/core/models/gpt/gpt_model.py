# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
import random


class GPTModel(LanguageModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        use_embedmix: bool = False,
        embedmix_subset_perturb: float = 0.1,
        embedmix_embedding_perturb: float = 0.1,
        embedmix_perturb_tokens_per_seq: float = 0.1,
        embedmix_augment_type: str = 'addition',
        embedmix_alpha: float = 0.1,

    ) -> None:
        super().__init__(config=config)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        #embedmix
        self.use_embedmix = use_embedmix
        self.embedmix_subset_perturb = embedmix_subset_perturb
        self.embedmix_embedding_perturb = embedmix_embedding_perturb
        self.embedmix_perturb_tokens_per_seq = embedmix_perturb_tokens_per_seq
        self.embedmix_augment_type = embedmix_augment_type
        self.embedmix_alpha = embedmix_alpha

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
            )

        if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            self.initialize_last_stage_with_word_embeddings()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])
    
    @staticmethod
    def embedmix_augment(
        dencoder_input: torch.Tensor,
        subset_perturb: float,
        embedding_perturb: float,
        perturb_tokens_per_seq: float,
        augment_type='addition',
        alpha: float = 0.5,
    ) -> torch.Tensor:

        """
        Data augmentation function embedmix which randomly swaps some parts of embeddings in a sequence

        Parameters:
            encoder_input : input embeddings, [x, y, z] x = tokens_quantity, y = batch_size, z = embedding_dim
            subset_perturb : which percent of sequences in a batch should be perturbed, belongs to [0, 1]
            embedding_perturb : which percent of single embedding should be perturbed, this is for embedding mask creation [0, 1], used for swap
            perturb_tokens_per_seq : which percent of embeddings in a single sequence should be perturbed [0, 1] 
        Returns:
            encoder_input: augmented_datapoints
        """
        if subset_perturb == 0:
            return dencoder_input
        dencoder_input = dencoder_input.clone() # making copy to avoid pytorch autograd problems
        batch_perturb_indeces = None
        if dencoder_input.size(1) == 1:  # batch size equals 1
            index = random.randint(
                0, 1
            )  # decided will the single sequence in a batch be augmented or not, E(X) ~ 0.5, Bernoulli distribution
            if index:
                batch_perturb_indeces = [0]
            else:
                return dencoder_input
        else:
            batch_perturb_indeces = random.sample(
                range(dencoder_input.size(1)), round(subset_perturb * dencoder_input.size(1))
            )
        for batch in batch_perturb_indeces:
            if augment_type == 'swap':
                tokens_perturb_quantity = round(dencoder_input.size(0) * perturb_tokens_per_seq)
            if augment_type == 'addition':
                tokens_perturb_quantity = round(dencoder_input.size(0) * perturb_tokens_per_seq) * 2 # to create pairs of tokens_quantity * perturb_tokens_per_seq 

            if tokens_perturb_quantity % 2 == 1:
                tokens_perturb_quantity += 1
            tokens_perturb_indices_pairs = torch.LongTensor(
                random.sample(range(dencoder_input.size(0)), tokens_perturb_quantity)
            ).reshape(-1, 2)
            for token_pair in tokens_perturb_indices_pairs:
                if augment_type == 'swap':
                    perturb_mask_len = round(dencoder_input.size(2) * embedding_perturb)
                    embedding_perturb_mask_start = random.randint(0, dencoder_input.size(2) - perturb_mask_len)
                    embedding_perturb_mask_end = embedding_perturb_mask_start + perturb_mask_len
                    (
                        dencoder_input[token_pair[0]][batch][embedding_perturb_mask_start:embedding_perturb_mask_end],
                        dencoder_input[token_pair[1]][batch][embedding_perturb_mask_start:embedding_perturb_mask_end],
                    ) = (
                        dencoder_input[token_pair[1]][batch][embedding_perturb_mask_start:embedding_perturb_mask_end],
                        dencoder_input[token_pair[0]][batch][embedding_perturb_mask_start:embedding_perturb_mask_end],
                    )
                if augment_type == 'addition':
                    vector_prime = dencoder_input[token_pair[1]][batch].clone()
                    dencoder_input[token_pair[0]][batch] += alpha * vector_prime

        return dencoder_input

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder

        if self.use_embedmix:
            decoder_input = self.embedmix_augment(
                dencoder_input = decoder_input,
                subset_perturb = self.embedmix_subset_perturb,
                embedding_perturb = self.embedmix_embedding_perturb ,
                perturb_tokens_per_seq = self.embedmix_perturb_tokens_per_seq,
                augment_type = self.embedmix_augment_type,
                alpha = self.embedmix_alpha
            )
        
        hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = ()) -> ShardedStateDict:
        assert not sharded_offsets, "Unexpected sharded offsets"
        sharded_state_dict = {}

        if self.pre_process:
            embedding_prefix = f'{prefix}embedding.'
            embedding_sharded_state_dict = self.embedding.sharded_state_dict(
                prefix=embedding_prefix
            )
            sharded_state_dict.update(embedding_sharded_state_dict)

        decoder_prefix = f'{prefix}decoder.'
        decoder_sharded_state_dict = self.decoder.sharded_state_dict(prefix=decoder_prefix)
        sharded_state_dict.update(decoder_sharded_state_dict)

        if self.post_process:
            output_layer_prefix = f'{prefix}output_layer.'
            output_layer_key = f'{output_layer_prefix}weight'
            if self.share_embeddings_and_output_weights:
                if not self.pre_process:
                    # when sharing embeddings with last stage, we need to use the weights from the first stage
                    # on pipeline first rank, word embeddings are saved to {prefix}embedding.word_embeddings.weight
                    tensor = self.shared_embedding_or_output_weight()
                    first_stage_word_emb_key = f'{prefix}embedding.word_embeddings.weight'
                    last_stage_word_emb_replica_id = (
                        1,  # copy of first stage embedding
                        0,
                        parallel_state.get_data_parallel_rank(with_context_parallel=True),
                    )

                    sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                        tensor=tensor,
                        key=first_stage_word_emb_key,
                        replica_id=last_stage_word_emb_replica_id,
                        allow_shape_mismatch=True,
                    )

                    sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

            else:
                output_layer_state_dict = self.output_layer.state_dict(
                    prefix=output_layer_prefix, keep_vars=True
                )
                output_layer_tensor = output_layer_state_dict[output_layer_key]
                # independent output layer
                sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                    tensor=output_layer_tensor, key=output_layer_key, allow_shape_mismatch=True,
                )

                sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

        return sharded_state_dict
