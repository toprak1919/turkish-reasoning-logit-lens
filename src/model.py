import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_hf_token

logger = logging.getLogger(__name__)


class LlamaWrapper:
    """Wrapper for Llama-3-8B with per-layer hidden state extraction and logit lens."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        dtype: str = "float16",
        device_map: str = "auto",
    ):
        token = get_hf_token()
        torch_dtype = getattr(torch, dtype)

        logger.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model: %s (dtype=%s)", model_name, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            token=token,
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.device = next(self.model.parameters()).device

        logger.info(
            "Model loaded: %d layers, hidden_dim=%d, vocab=%d",
            self.n_layers, self.hidden_dim, self.vocab_size,
        )

    @torch.no_grad()
    def get_all_hidden_states(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass returning hidden states at all layers.

        Returns:
            Tuple of (n_layers + 1) tensors, each (batch, seq_len, hidden_dim).
            Index 0 = embedding output, index i = output of transformer layer i.
        """
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids, output_hidden_states=True)
        return outputs.hidden_states

    @torch.no_grad()
    def decode_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Project a hidden state through the final RMSNorm and unembedding head.

        CRITICAL: Llama-3's final LayerNorm (RMSNorm) sits between the last
        transformer block and the lm_head. Skipping it produces meaningless logits.

        Args:
            hidden_state: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)

        Returns:
            logits: same leading dims + (vocab_size,)
        """
        normed = self.model.model.norm(hidden_state)
        logits = self.model.lm_head(normed)
        return logits

    @torch.no_grad()
    def get_logit_lens_all_layers(
        self, input_ids: torch.Tensor, position: int = -1
    ) -> torch.Tensor:
        """
        Compute logit lens at all layers for a specific token position.

        Args:
            input_ids: (1, seq_len) tokenized input
            position: which token position to analyze (-1 = last token)

        Returns:
            all_logits: (n_layers + 1, vocab_size) logits at each layer
        """
        hidden_states = self.get_all_hidden_states(input_ids)
        all_logits = []
        for hs in hidden_states:
            h = hs[0, position, :]  # (hidden_dim,)
            logits = self.decode_hidden_state(h.unsqueeze(0).unsqueeze(0))
            all_logits.append(logits.squeeze())
        # Free the full hidden states tuple to reduce GPU memory pressure
        del hidden_states
        return torch.stack(all_logits)  # (n_layers+1, vocab_size)

    def tokenize(self, text: str, return_tensors: str = "pt") -> dict:
        return self.tokenizer(text, return_tensors=return_tensors)

    def get_answer_token_ids(self, answer_number: int) -> list[int]:
        """
        Get plausible token IDs for a numeric answer.
        Checks multiple tokenization contexts since BPE is context-dependent.
        """
        answer_str = str(answer_number)
        candidates = set()

        # Standalone
        ids = self.tokenizer.encode(answer_str, add_special_tokens=False)
        candidates.add(ids[0])

        # With leading space (common in generation context)
        ids = self.tokenizer.encode(f" {answer_str}", add_special_tokens=False)
        candidates.add(ids[0])

        # After newline
        ids = self.tokenizer.encode(f"\n{answer_str}", add_special_tokens=False)
        if len(ids) > 1:
            candidates.add(ids[1])
        else:
            candidates.add(ids[0])

        return list(candidates)

    def get_tokenization_info(self, text_en: str, text_tr: str) -> dict:
        """Compare tokenization of English and Turkish texts."""
        tokens_en = self.tokenizer.encode(text_en, add_special_tokens=False)
        tokens_tr = self.tokenizer.encode(text_tr, add_special_tokens=False)
        return {
            "tokens_en": tokens_en,
            "tokens_tr": tokens_tr,
            "decoded_en": [self.tokenizer.decode([t]) for t in tokens_en],
            "decoded_tr": [self.tokenizer.decode([t]) for t in tokens_tr],
            "n_tokens_en": len(tokens_en),
            "n_tokens_tr": len(tokens_tr),
            "fertility_ratio": len(tokens_tr) / max(len(tokens_en), 1),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from a prompt (for behavioral evaluation)."""
        inputs = self.tokenize(prompt)
        input_ids = inputs["input_ids"].to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        output_ids = self.model.generate(input_ids, **gen_kwargs)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
