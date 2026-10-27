"""Общие фикстуры. Ничего не скачивается: модели собираются из конфига со
случайными весами, синтетический токенизатор обучается на лету.

Реальный токенизатор подключается через переменную окружения:

    STF_TOKENIZER=~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit pytest

Для механики хуков обученные веса не нужны: важны раскладка слоёв, формат
выхода блока, позиции на prefill/decode и KV-кэш — это свойства архитектуры.
"""

from __future__ import annotations

import os

import pytest

# Шаблон в стиле Qwen: ответ инструмента внутри user-хода в <tool_response>.
CHATML_TOOL_TEMPLATE = r"""
{%- for m in messages -%}
{%- if m.role == 'tool' -%}
{{- '<|im_start|>user\n<tool_response>\n' + m.content + '\n</tool_response><|im_end|>\n' -}}
{%- elif m.role == 'assistant' and m.tool_calls -%}
{{- '<|im_start|>assistant\n' -}}
{%- for tc in m.tool_calls -%}
{{- '<tool_call>\n' + (tc.function | tojson) + '\n</tool_call>' -}}
{%- endfor -%}
{{- '<|im_end|>\n' -}}
{%- else -%}
{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}
"""


@pytest.fixture(scope="session")
def chatml_tokenizer():
    """Маленький byte-level BPE + ChatML-шаблон с ролью tool."""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    return make_chatml_tokenizer()


def make_chatml_tokenizer():
    import tokenizers
    from tokenizers import decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from steering_tool_feedback import simulator as S

    specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tk = tokenizers.Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = []
    for t in S.build_pilot_set(60, seed=1):
        for b in (0, 1):
            corpus += [m.get("content", "") for m in t.prompt(b)]
    corpus += ["<tool_response>", "</tool_response>", "<tool_call>", "</tool_call>",
               "assistant", "user", "system"]
    # имена действий — целыми токенами, как у настоящих токенизаторов (у всех разный первый токен)
    corpus += [f'{{"name": "{a}"}}' for a in S.ACTIONS] * 20
    tk.train_from_iterator(corpus * 3, trainer)

    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = CHATML_TOOL_TEMPLATE
    return tok


@pytest.fixture(scope="session")
def real_tokenizer():
    path = os.environ.get("STF_TOKENIZER")
    if not path:
        pytest.skip("set STF_TOKENIZER=<local tokenizer dir or HF id> to run on a real template")
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.expanduser(path))


# --------------------------------------------------------------------------
# Модели из конфига
# --------------------------------------------------------------------------


def make_tiny_llama(vocab: int = 512):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
    )
    return LlamaForCausalLM(cfg).eval()


def make_tiny_qwen35(vocab: int = 512):
    """Та же архитектура, что у Qwen3.x: 3 слоя линейного внимания + 1 полного."""
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        max_position_embeddings=4096,
    )
    return Qwen3_5ForCausalLM(cfg).eval()


MODEL_FACTORIES = {"llama": make_tiny_llama, "qwen3_5": make_tiny_qwen35}


@pytest.fixture(scope="session", params=sorted(MODEL_FACTORIES))
def tiny_model(request):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        return request.param, MODEL_FACTORIES[request.param]()
    except ImportError as e:
        pytest.skip(f"{request.param} not available in this transformers: {e}")
