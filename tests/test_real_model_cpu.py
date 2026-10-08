"""Exercises the real transformers and peft code paths on CPU with a tiny random model.
No download. Checks prompt masking, LoRA training, adapter load and unload, and truncation."""
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishara.backend import HFBackend, build_messages  # noqa: E402
from ishara.lora import encode_pair, train_lora  # noqa: E402

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def tiny():
    chars = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!'")
    specials = ["<pad>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<unk>"]
    vocab = {t: i for i, t in enumerate(specials + chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tk.add_special_tokens(specials)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = TEMPLATE
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      eos_token_id=[vocab["<|end|>"], vocab["<eos>"]], pad_token_id=0)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = [vocab["<|end|>"], vocab["<eos>"]]
    return tok, model


def make_backend(max_new_tokens=8):
    tok, model = tiny()
    b = HFBackend.__new__(HFBackend)
    b.torch = torch
    b.tok, b.model, b.base_model, b.adapter = tok, model, model, None
    b.mcfg = {"chat_template_kwargs": {}, "load_in_4bit": False}
    b.gcfg = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": 0.7, "top_p": 0.9}
    b.eos_ids = set(model.generation_config.eos_token_id) | {tok.eos_token_id}
    b.device_name = "cpu-test"
    return b


def test_encode_pair_masks_prompt():
    tok, _ = tiny()
    msgs = build_messages("kya haal hai", "be kind")
    ids, labels = encode_pair(tok, msgs, "theek", 512)
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids["input_ids"])
    n = len(prompt_ids)
    assert ids[:n] == prompt_ids
    assert all(x == -100 for x in labels[:n])
    resp = tok.decode([t for t in labels[n:] if t != -100], skip_special_tokens=True)
    assert resp.replace(" ", "") == "theek"
    assert labels[-1] == tok.convert_tokens_to_ids("<|end|>")


def test_generate_seeded_and_truncation():
    b = make_backend(max_new_tokens=8)
    msgs = build_messages("hello there")
    r1, _ = b.generate(msgs, 3, 0, "k1")
    r2, _ = b.generate(msgs, 3, 0, "k1")
    assert [x["response"] for x in r1] == [x["response"] for x in r2]
    for x in r1:
        assert x["truncated"] == (x["n_new_tokens"] >= 8)


def test_lora_train_load_unload():
    b = make_backend()
    before = {k: v.clone() for k, v in b.base_model.state_dict().items()}
    pairs = [(build_messages("please help"), "sure")] * 4
    acfg = {"lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "lr": 1e-2, "epochs": 2, "grad_accum": 2,
            "max_len": 256, "gradient_checkpointing": True}
    with tempfile.TemporaryDirectory() as d:
        secs, losses = train_lora(b, pairs, 0, acfg, Path(d) / "adapter")
        assert (Path(d) / "adapter" / "adapter_config.json").exists()
        assert losses[-1] < losses[0]
        after = b.base_model.state_dict()
        assert set(before) == set(after)
        for k in before:
            assert torch.equal(before[k], after[k]), k
        msgs = build_messages("please help")
        base_out, _ = b.generate(msgs, 1, 0, "x")
        b.load_adapter(Path(d) / "adapter")
        assert b.adapter is not None
        ad_out, _ = b.generate(msgs, 1, 0, "x")
        b.clear_adapter()
        base_again, _ = b.generate(msgs, 1, 0, "x")
        assert base_out[0]["response"] == base_again[0]["response"]
        for k, v in b.base_model.state_dict().items():
            assert torch.equal(before[k], v), k


if __name__ == "__main__":
    test_encode_pair_masks_prompt()
    test_generate_seeded_and_truncation()
    test_lora_train_load_unload()
    print("real model CPU tests passed")
