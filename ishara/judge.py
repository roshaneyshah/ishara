"""LLM judge for the A2 feedback loop only. Validate it against human scores before use."""
import json
import re

JUDGE_TEMPLATE = """You are scoring a reply from an assistant. Use the rubric below.

RUBRIC
{rubric}

USER MESSAGE
{prompt}
{reference_block}
ASSISTANT REPLY
{response}

Score each dimension 0, 1 or 2. Answer with one JSON object and nothing else, using these keys:
{keys}"""


def parse_scores(text, dims, lo=0, hi=2):
    """Take the last JSON object in the text. Returns dict or None."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text or "", flags=re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        try:
            out = {d: int(obj[d]) for d in dims}
        except (KeyError, TypeError, ValueError):
            continue
        if all(lo <= v <= hi for v in out.values()):
            return out
    return None


class Judge:
    def __init__(self, backend, rubric_text, dims, threshold):
        self.backend, self.rubric, self.dims, self.threshold = backend, rubric_text, dims, threshold
        self.parse_failures = 0

    def score(self, prompt, response, reference="", key="judge"):
        ref = f"\nWHAT THE USER MEANT\n{reference}\n" if reference else ""
        text = JUDGE_TEMPLATE.format(rubric=self.rubric, prompt=prompt, reference_block=ref,
                                     response=response, keys=", ".join(self.dims))
        msgs = [{"role": "user", "content": text}]
        out, _ = self.backend.generate(msgs, 1, 0, key)
        s = parse_scores(out[0]["response"], self.dims)
        if s is None:
            self.parse_failures += 1
        return s

    def thumbs(self, prompt, response, reference="", key="judge"):
        s = self.score(prompt, response, reference, key)
        if s is None:
            return False, None
        return sum(s.values()) >= self.threshold, s


class HumanOracle:
    """Thumbs typed in by a person in the notebook. Used when a2.oracle is human."""
    parse_failures = 0

    def thumbs(self, prompt, response, reference="", key=""):
        print("\n" + "=" * 60 + "\nPROMPT:\n" + prompt + "\n\nRESPONSE:\n" + response)
        while True:
            a = input("up or down? ").strip().lower()
            if a in ("up", "down"):
                return a == "up", None
