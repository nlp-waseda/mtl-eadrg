import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart import (
    PretrainedBartModel,
    _make_linear_from_emb,
    _reorder_buffer,
    BartClassificationHead,
    BartModel
)


class BartForMultitaskLearning(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.model = BartModel(config)
        self.register_buffer(
            "final_logits_bias",
            torch.zeros((1, self.model.shared.num_embeddings))
        )

        self.num_cfemotions = 12
        self.num_emotions = 6
        self.num_sentiments = 2

        self.cfemotion_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_cfemotions,
            config.classif_dropout
        )
        self.model._init_weights(self.cfemotion_head.dense)
        self.model._init_weights(self.cfemotion_head.out_proj)

        self.emotion_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_emotions,
            config.classif_dropout
        )
        self.model._init_weights(self.emotion_head.dense)
        self.model._init_weights(self.emotion_head.out_proj)

        self.sentiment_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_sentiments,
            config.classif_dropout
        )
        self.model._init_weights(self.sentiment_head.dense)
        self.model._init_weights(self.sentiment_head.out_proj)

    def resize_token_embeddings(self, new_num_tokens):
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens, old_num_tokens):
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens),
                device=self.final_logits_bias.device
            )
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        task=None,
        **unused
    ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed "
                "in a future version, use `labels` instead.",
                DeprecationWarning
            )
            labels = unused.pop("lm_labels")

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        if task == "response":
            lm_logits = F.linear(
                outputs[0],
                self.model.shared.weight,
                bias=self.final_logits_bias
            )
            outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # TODO(SS): do we need to ignore pad tokens in labels?
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size),
                    labels.view(-1)
                )
                outputs = (masked_lm_loss,) + outputs

        elif task in ["cfemotion", "emotion", "sentiment"]:
            x = outputs[0]  # last hidden state

            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError(
                   "All examples must have the same number of <eos> tokens."
                )

            if task == "cfemotion":
                classification_head = self.cfemotion_head
                num_labels = self.num_cfemotions
            elif task == "emotion":
                classification_head = self.emotion_head
                num_labels = self.num_emotions
            else:
                classification_head = self.sentiment_head
                num_labels = self.num_sentiments

            sentence_representation = x[eos_mask, :].view(
                x.size(0),
                -1,
                x.size(-1)
            )[:, -1, :]
            logits = classification_head(sentence_representation)

            # Prepend logits
            outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
            if labels is not None:  # prepend loss to output,
                loss = F.cross_entropy(
                    logits.view(-1, num_labels),
                    labels.view(-1)
                )
                outputs = (loss,) + outputs
        
        else:
            raise ValueError("The dataset contains an invalid task.")

        return outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past,
        attention_mask,
        use_cache,
        task,
        **kwargs
    ):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "task": task
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device
        )
        assert len(scores.shape) == 2, \
            "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx)
                for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = (
            enc_out if enc_out is None 
            else enc_out.index_select(0, beam_idx)
        )
        new_enc_mask = (
            enc_mask if enc_mask is None
            else enc_mask.index_select(0, beam_idx)
        )

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly


class BartForAdversarialMultitaskLearning(BartForMultitaskLearning):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        task=None,
        **unused
    ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed "
                "in a future version, use `labels` instead.",
                DeprecationWarning
            )
            labels = unused.pop("lm_labels")

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        if task == "response":
            lm_logits = F.linear(
                outputs[0],
                self.model.shared.weight,
                bias=self.final_logits_bias
            )
            outputs = (lm_logits,) + outputs#[1:]  # Add cache, hidden states and attention if they are here

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # TODO(SS): do we need to ignore pad tokens in labels?
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size),
                    labels.view(-1)
                )
                outputs = (masked_lm_loss,) + outputs

        elif task in ["cfemotion", "emotion", "sentiment"]:
            x = outputs[0]  # last hidden state

            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError(
                   "All examples must have the same number of <eos> tokens."
                )

            if task == "cfemotion":
                classification_head = self.cfemotion_head
                num_labels = self.num_cfemotions
            elif task == "emotion":
                classification_head = self.emotion_head
                num_labels = self.num_emotions
            else:
                classification_head = self.sentiment_head
                num_labels = self.num_sentiments

            sentence_representation = x[eos_mask, :].view(
                x.size(0),
                -1,
                x.size(-1)
            )[:, -1, :]
            logits = classification_head(sentence_representation)

            # Prepend logits
            outputs = (logits,) + outputs#[1:]  # Add hidden states and attention if they are here
            if labels is not None:  # prepend loss to output,
                loss = F.cross_entropy(
                    logits.view(-1, num_labels),
                    labels.view(-1)
                )
                outputs = (loss,) + outputs
        
        else:
            raise ValueError("The dataset contains an invalid task.")

        return outputs
