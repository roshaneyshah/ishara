"""Model backends. HFBackend runs the real model on a GPU. FakeBackend is for tests only."""
import json
import time
from pathlib import Path

from .util import stable_int


def build_messages(user_prompt, system=None, demos=()):
    """Chat messages: optional system prompt, then demonstration pairs, then the user prompt."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for d in demos:
        msgs.append({"role": "user", "content": d["prompt"]})
        msgs.append({"role": "assistant", "content": d["response"]})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. In Colab: Runtime > Change runtime type > T4 GPU. In Kaggle: "
                           "Settings > Accelerator > GPU T4. A CPU runtime offloads the model and stalls.")
    return torch.cuda.get_device_name(0)


class HFBackend:
    def __init__(self, mcfg, gcfg):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not mcfg.get("hf_id") or "REPLACE" in str(mcfg["hf_id"]):
            raise ValueError("Set model.hf_id in the config.")
        if not mcfg.get("revision"):
            raise ValueError("Pin model.revision to a commit hash.")
        self.device_name = require_gpu()
        self.torch = torch
        self.mcfg, self.gcfg = mcfg, gcfg
        dtype = getattr(torch, mcfg.get("dtype", "float16"))
        kw = dict(revision=mcfg["revision"], trust_remote_code=mcfg.get("trust_remote_code", False))
        self.tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], **kw)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        load_kw = dict(kw)
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        load_kw[dtype_key] = dtype
        if mcfg.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
            load_kw["device_map"] = {"": 0}
            self.model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], **load_kw)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], **load_kw).to("cuda")
        off = [n for n, p in self.model.named_parameters() if p.device.type != "cuda"]
        if off:
            raise RuntimeError(f"{len(off)} parameters are not on the GPU, for example {off[0]}.")
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.eos_ids = set(e for e in eos if e is not None) | {self.tok.eos_token_id}
        self.base_model = self.model
        self.adapter = None

    # chat template with a fallback for templates that reject the system role
    def _encode(self, messages):
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True,
                  **(self.mcfg.get("chat_template_kwargs") or {}))
        try:
            enc = self.tok.apply_chat_template(messages, **kw)
        except Exception:
            if messages and messages[0]["role"] == "system":
                sys_text = messages[0]["content"]
                rest = [dict(m) for m in messages[1:]]
                rest[0]["content"] = sys_text + "\n\n" + rest[0]["content"]
                enc = self.tok.apply_chat_template(rest, **kw)
            else:
                raise
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def generate(self, messages, n, seed, key):
        """n samples for one prompt. Seeded from (seed, key) so reruns match."""
        torch = self.torch
        g = self.gcfg
        enc = self._encode(messages)
        torch.manual_seed(stable_int(seed, key))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stable_int(seed, key))
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=g["max_new_tokens"], do_sample=g["do_sample"],
                temperature=g["temperature"] if g["do_sample"] else None,
                top_p=g["top_p"] if g["do_sample"] else None,
                num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        secs = time.time() - t0
        new = out[:, enc["input_ids"].shape[1]:].tolist()
        results = []
        for row in new:
            cut = next((i for i, t in enumerate(row) if t in self.eos_ids), None)
            if cut is None:
                toks, truncated = row, len(row) >= g["max_new_tokens"]
            else:
                toks, truncated = row[:cut], False
            results.append({"response": self.tok.decode(toks, skip_special_tokens=True).strip(),
                            "n_new_tokens": len(toks), "truncated": truncated})
        return results, secs

    def train_lora(self, pairs, seed, acfg, out_path):
        from .lora import train_lora
        return train_lora(self, pairs, seed, acfg, out_path)

    def load_adapter(self, path):
        from peft import PeftModel
        self.clear_adapter()
        self.model = PeftModel.from_pretrained(self.base_model, path)
        self.model.eval()
        self.adapter = str(path)

    def clear_adapter(self):
        if self.adapter is not None:
            self.base_model = self.model.unload()
            self.model = self.base_model
            self.adapter = None


class FakeBackend:
    """Deterministic stand-in used by the tests. It never touches a GPU."""

    def __init__(self, max_new_tokens=1024, judge_dims=None):
        self.max_new_tokens = max_new_tokens
        self.adapter = None
        self.device_name = "fake"
        self.judge_dims = judge_dims

    def generate(self, messages, n, seed, key):
        if self.judge_dims:
            h = stable_int(seed, key, messages[-1]["content"])
            obj = {d: (h >> (2 * i)) % 3 for i, d in enumerate(self.judge_dims)}
            return [{"response": json.dumps(obj), "n_new_tokens": 20, "truncated": False}], 0.01
        sys_text = messages[0]["content"] if messages[0]["role"] == "system" else ""
        n_demos = sum(1 for m in messages if m["role"] == "assistant")
        res = []
        for i in range(n):
            h = stable_int(seed, key, i, sys_text, n_demos, self.adapter)
            truncated = h % 17 == 0
            res.append({"response": f"fake reply {h % 1000} demos={n_demos} adapter={self.adapter}",
                        "n_new_tokens": self.max_new_tokens if truncated else 50 + h % 200,
                        "truncated": truncated})
        return res, 0.01 * n

    def train_lora(self, pairs, seed, acfg, out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)
        (Path(out_path) / "adapter_config.json").write_text(json.dumps({"fake": True, "n": len(pairs)}))
        return 0.02 * len(pairs), [1.0] * len(pairs)

    def load_adapter(self, path):
        self.adapter = str(path)

    def clear_adapter(self):
        self.adapter = None
