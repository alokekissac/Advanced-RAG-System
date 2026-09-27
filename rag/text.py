"""Tokenisation shared by the lexical components."""
import re

STOPWORDS = frozenset("""a an and are as at be been but by can do does for from has have how i if in into is it its
may me must my no not of on or our so that the their them then there these they this to up us was we were what
when where which who why will with you your i'm i've can't don't get got many much long often need needs
allowed allow tell know there's does""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def stem(t: str) -> str:
    """Light suffix stripping so 'employs', 'employed' and 'employ' match."""
    if len(t) <= 3 or not t.isalpha():
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and t[:-2].endswith(("ss", "x", "z", "ch", "sh")):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    for suf in ("ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "lsz":   # shipping -> ship
                t = t[:-1]
            break
    if len(t) >= 4 and t.endswith("e") and not t.endswith("ee"):   # approve / approved -> approv
        t = t[:-1]
    return t


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [stem(t) for t in toks if keep_stopwords or t not in STOPWORDS]
