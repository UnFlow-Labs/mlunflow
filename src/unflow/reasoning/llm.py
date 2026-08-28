from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import transformers
from llama_cpp import Llama


@dataclass(slots=True)
class LLMConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.0
    extra_generate_kwargs: dict[str, Any] = field(default_factory=dict)
    


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: LLMConfig | None = None) -> str:
        raise NotImplementedError

class LlamaLLM(BaseLLM):
    def __init__(self, repo_id: str, filename: str = "*.gguf", n_ctx: int | None = None):
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        try:
            model_kwargs: dict[str, Any] = {"verbose": False}
            if n_ctx is not None:
                model_kwargs["n_ctx"] = n_ctx
            self.model = Llama.from_pretrained(repo_id=repo_id, filename=filename, **model_kwargs)
        except ValueError as exc:
            message = str(exc)
            if "No file found" in message and ".gguf" in filename:
                raise ValueError(
                    f"{message}\n\n"
                    "This repo appears to provide Transformers weights (e.g. safetensors), not GGUF. "
                    "Use TransformersLLM(model_name=repo_id) for this model, or pass a repo/filename that contains GGUF files."
                ) from exc
            raise

    def generate(self, prompt: str, config: LLMConfig | None = None) -> str:
        cfg = config or LLMConfig()
        context_window = self.n_ctx or 512
        n_ctx_attr = getattr(self.model, "n_ctx", None)
        if callable(n_ctx_attr):
            model_n_ctx = int(n_ctx_attr())
            if model_n_ctx > 0:
                context_window = model_n_ctx
        elif isinstance(n_ctx_attr, int) and n_ctx_attr > 0:
            context_window = n_ctx_attr

        prompt_tokens = self.model.tokenize(prompt.encode("utf-8"))
        requested_max_tokens = max(1, int(cfg.max_new_tokens))

        if len(prompt_tokens) + requested_max_tokens > context_window:
            max_prompt_tokens = max(1, context_window - requested_max_tokens)
            prompt_tokens = prompt_tokens[-max_prompt_tokens:]
            prompt = self.model.detokenize(prompt_tokens).decode("utf-8", errors="ignore") or prompt
            prompt_tokens = self.model.tokenize(prompt.encode("utf-8"))

        if len(prompt_tokens) >= context_window:
            prompt_tokens = prompt_tokens[-max(1, context_window - 1) :]
            prompt = self.model.detokenize(prompt_tokens).decode("utf-8", errors="ignore")
            prompt_tokens = self.model.tokenize(prompt.encode("utf-8")) if prompt else []

        max_tokens = min(requested_max_tokens, max(1, context_window - len(prompt_tokens)))

        def _extract_text(raw_response: Any) -> str:
            choices = raw_response.get("choices", []) if isinstance(raw_response, dict) else []
            first_choice = choices[0] if choices else {}

            if isinstance(first_choice, dict):
                text = str(first_choice.get("text", "")).strip()
                if text:
                    return text

                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = str(message.get("content", "")).strip()
                    if content:
                        return content

            return ""
        
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repeat_penalty=cfg.repetition_penalty,
            **cfg.extra_generate_kwargs,
        )
        text = _extract_text(response)

        if text:
            return text

        try:
            chat_response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repetition_penalty,
                **cfg.extra_generate_kwargs,
            )
            chat_text = _extract_text(chat_response)
            if chat_text:
                return chat_text
        except Exception:
            pass

        fallback = str(response).strip()
        return fallback

class TransformersLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, config: LLMConfig | None = None) -> str:
        cfg = config or LLMConfig()

        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            chat_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(chat_inputs, "input_ids"):
                input_ids = chat_inputs.input_ids
                attention_mask = getattr(chat_inputs, "attention_mask", None)
            elif isinstance(chat_inputs, dict):
                input_ids = chat_inputs["input_ids"]
                attention_mask = chat_inputs.get("attention_mask")
            else:
                input_ids = chat_inputs
                attention_mask = None

            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)

            if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long, device=self.device)
                if attention_mask.ndim == 1:
                    attention_mask = attention_mask.unsqueeze(0)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **cfg.extra_generate_kwargs,
        ) #type: ignore

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        text = decoded.strip() if isinstance(decoded, str) else str(decoded).strip()
        if text:
            return text

        raw_decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        raw_text = raw_decoded.strip() if isinstance(raw_decoded, str) else str(raw_decoded).strip()
        return raw_text


