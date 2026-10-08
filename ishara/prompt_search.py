"""A2: evolutionary search over system prompts built from a fixed fragment library.

Budget accounting: every output shown to the oracle costs one signal. The search stops when the
budget is spent, so the number of signals used equals the budget exactly.
"""
import random

from .backend import build_messages


def load_fragments(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def render(genome, fragments):
    return "\n".join(fragments[i] for i in genome) if genome else None


def _posterior(stat):
    ups, n = stat
    return (ups + 1) / (n + 2)


def search(backend, oracle, pool_df, fragments, budget, seed, a2cfg, log):
    """Returns (best_genome, stats). `log` is a list that receives one dict per signal."""
    rng = random.Random(seed)
    nf = len(fragments)
    maxf = min(a2cfg["max_fragments"], nf)
    pop_size = a2cfg["population"]
    batch = a2cfg["eval_batch"]
    prompts = list(pool_df.itertuples())
    stats = {}
    remaining = [budget]
    counter = [0]

    def evaluate(genome):
        m = min(batch, remaining[0])
        if m <= 0:
            return
        rows = rng.sample(prompts, m) if m <= len(prompts) else [rng.choice(prompts) for _ in range(m)]
        system = render(genome, fragments)
        ups = 0
        for r in rows:
            counter[0] += 1
            key = f"a2|{seed}|{budget}|{counter[0]}|{r.prompt_id}"
            out, secs = backend.generate(build_messages(r.prompt, system), 1, seed, key)
            up, scores = oracle.thumbs(r.prompt, out[0]["response"], r.reference, key)
            ups += int(up)
            log.append({"signal": counter[0], "genome": list(genome), "prompt_id": r.prompt_id,
                        "response": out[0]["response"], "truncated": out[0]["truncated"],
                        "up": bool(up), "judge_scores": scores, "gen_secs": secs})
        u, n = stats.get(genome, (0, 0))
        stats[genome] = (u + ups, n + m)
        remaining[0] -= m

    def random_genome():
        k = rng.randint(1, maxf)
        g = rng.sample(range(nf), k)
        return tuple(g)

    def mutate(g):
        g = list(g)
        ops = ["add", "remove", "swap", "replace"]
        op = rng.choice(ops)
        unused = [i for i in range(nf) if i not in g]
        if op == "add" and len(g) < maxf and unused:
            g.insert(rng.randint(0, len(g)), rng.choice(unused))
        elif op == "remove" and g:
            g.pop(rng.randrange(len(g)))
        elif op == "swap" and len(g) >= 2:
            i, j = rng.sample(range(len(g)), 2)
            g[i], g[j] = g[j], g[i]
        elif unused and g:
            g[rng.randrange(len(g))] = rng.choice(unused)
        elif unused:
            g.append(rng.choice(unused))
        return tuple(g)

    def crossover(a, b):
        cut_a = rng.randint(0, len(a))
        cut_b = rng.randint(0, len(b))
        child = []
        for i in list(a[:cut_a]) + list(b[cut_b:]):
            if i not in child:
                child.append(i)
        return tuple(child[:maxf])

    population = [()]
    while len(population) < pop_size:
        g = random_genome()
        if g not in population:
            population.append(g)
    for g in population:
        evaluate(g)

    while remaining[0] > 0:
        ranked = sorted(population, key=lambda g: -_posterior(stats.get(g, (0, 0))))
        p1 = max(rng.sample(ranked, min(2, len(ranked))), key=lambda g: _posterior(stats.get(g, (0, 0))))
        p2 = max(rng.sample(ranked, min(2, len(ranked))), key=lambda g: _posterior(stats.get(g, (0, 0))))
        child = mutate(crossover(p1, p2)) if rng.random() < 0.5 else mutate(p1)
        evaluate(child)
        pool = set(population) | {child}
        population = sorted(pool, key=lambda g: -_posterior(stats.get(g, (0, 0))))[:pop_size]

    evaluated = [g for g in stats]
    best = max(evaluated, key=lambda g: (_posterior(stats[g]), stats[g][1], -len(g)))
    used = sum(n for _, n in stats.values())
    assert used == budget, (used, budget)
    return best, {"|".join(map(str, g)): {"ups": u, "n": n} for g, (u, n) in stats.items()}
