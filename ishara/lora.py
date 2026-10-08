"""A3 and A4: LoRA fine-tuning on (prompt, response) pairs. Loss on response tokens only."""
import random
import time


def encode_pair(tok, messages, response, max_len, template_kwargs=None):
    """Token ids and labels with the prompt part masked. Uses the chat template when the prompt
    encoding is a prefix of the full encoding, otherwise appends the response and eos by hand."""
    kw = dict(template_kwargs or {})
    p_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kw)
    if isinstance(p_ids, dict) or hasattr(p_ids, "input_ids"):
        p_ids = p_ids["input_ids"]
    p_ids = list(p_ids)
    full = messages + [{"role": "assistant", "content": response}]
    try:
        f_ids = tok.apply_chat_template(full, tokenize=True, **kw)
        if isinstance(f_ids, dict) or hasattr(f_ids, "input_ids"):
            f_ids = f_ids["input_ids"]
        f_ids = list(f_ids)
    except Exception:
        f_ids = None
    if f_ids is None or f_ids[:len(p_ids)] != p_ids or len(f_ids) <= len(p_ids):
        f_ids = p_ids + tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    f_ids, labels = f_ids[:max_len], labels[:max_len]
    return f_ids, labels


def train_lora(backend, pairs, seed, acfg, out_path):
    """pairs: list of (messages, response). Trains a fresh adapter on backend.base_model,
    saves it to out_path and restores the base model. Returns training seconds."""
    import torch
    from peft import LoraConfig, get_peft_model

    backend.clear_adapter()
    model = backend.base_model
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if backend.mcfg.get("load_in_4bit"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=acfg.get("gradient_checkpointing", True))
    elif acfg.get("gradient_checkpointing", True):
        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        except (ValueError, NotImplementedError, AttributeError) as e:
            print(f"Gradient checkpointing not supported by this model ({e}). Training without it.")
    model.config.use_cache = False
    lcfg = LoraConfig(r=acfg["lora_r"], lora_alpha=acfg["lora_alpha"], lora_dropout=acfg["lora_dropout"],
                      target_modules="all-linear", task_type="CAUSAL_LM")
    pm = get_peft_model(model, lcfg)
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    tkw = backend.mcfg.get("chat_template_kwargs") or {}
    data = [encode_pair(backend.tok, m, r, acfg["max_len"], tkw) for m, r in pairs]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=acfg["lr"])
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    dev = next(pm.parameters()).device
    rng = random.Random(seed)
    accum = acfg["grad_accum"]
    pm.train()
    t0 = time.time()
    step_in_accum = 0
    losses = []
    for _ in range(acfg["epochs"]):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in order:
            ids, labels = data[i]
            ids_t = torch.tensor([ids], device=dev)
            lab_t = torch.tensor([labels], device=dev)
            with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                dtype=torch.float16 if use_cuda else torch.bfloat16, enabled=use_cuda):
                loss = pm(input_ids=ids_t, labels=lab_t).loss
            losses.append(float(loss.detach()))
            scaler.scale(loss / accum).backward()
            step_in_accum += 1
            if step_in_accum == accum:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step_in_accum = 0
    if step_in_accum:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    if use_cuda:
        torch.cuda.synchronize()
    secs = time.time() - t0
    pm.save_pretrained(out_path)
    base = pm.unload()
    if acfg.get("gradient_checkpointing", True) and hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    if hasattr(base, "disable_input_require_grads"):
        try:
            base.disable_input_require_grads()
        except AttributeError:
            pass
    base.config.use_cache = True
    base.eval()
    backend.base_model = base
    backend.model = base
    return secs, losses
