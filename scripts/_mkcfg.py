"""Material config z bazy + nadpisania `klucz.sciezka=wartosc`. Do sweepu.

Po co osobny skrypt, a nie flagi w train_cl: agent W&B podaje parametry jako argumenty procesu,
a my potrzebujemy z nich PLIKU configu, bo to config ladzie w run-info.txt i to on jest
proweniencja wyniku. Bez zapisanego pliku punkt sweepu bylby nieodtwarzalny.

Typy bierzemy z wartosci w bazie: jesli tam stoi float, wartosc jest rzutowana na float, jesli
bool -- na bool. Dzieki temu "reg.weight=500" nie wpisze stringa tam, gdzie reszta kodu czeka
liczby. Klucz, ktorego w bazie nie ma, musi dostac jawny typ (`:int`, `:float`, `:bool`),
inaczej skrypt przerywa -- ciche wpisanie stringa to dokladnie ten rodzaj bledu, ktory ujawnia
sie trzy godziny pozniej.

Run:  python scripts/_mkcfg.py --base configs/phaseT/T50_mixed.yaml \
          --out configs/sweep/p001.yaml --set reg.weight=500 reg.space=dw task_cond.key_dim=256
"""
import argparse
import os
import re


def _typed(raw, old):
    if raw.endswith((":int", ":float", ":bool", ":str")):
        raw, t = raw.rsplit(":", 1)
        return {"int": int, "float": float, "str": str,
                "bool": lambda v: v.lower() in ("1", "true", "yes")}[t](raw)
    if old is None:
        raise SystemExit(f"klucza nie ma w bazie i nie podano typu: {raw!r} "
                         "(dopisz :int, :float, :bool albo :str)")
    if re.fullmatch(r"(true|false)", old, re.I):
        return raw.lower() in ("1", "true", "yes")
    if re.fullmatch(r"-?\d+", old):
        return int(raw)
    if re.fullmatch(r"-?[\d.]+([eE][-+]?\d+)?", old):
        return float(raw)
    return raw


def _section_end(lines, parts):
    """Indeks PIERWSZEJ linii za sekcja `parts` i wciecie jej kluczy, albo None."""
    depth, i = 0, 0
    while i < len(lines):
        m = re.match(r"^(\s*)([A-Za-z_][\w]*):(.*)$", lines[i])
        if m and len(m.group(1)) == depth * 2 and m.group(2) == parts[depth]:
            depth += 1
            if depth == len(parts):
                ind = depth * 2
                j = i + 1
                while j < len(lines) and (not lines[j].strip()
                                          or lines[j].startswith(" " * ind)):
                    j += 1
                return j, ind
        i += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="SCIEZKA=WARTOSC")
    a = ap.parse_args()

    lines = open(a.base, encoding="utf-8").read().split("\n")
    name = os.path.splitext(os.path.basename(a.out))[0]

    for item in a.set:
        path, _, raw = item.partition("=")
        parts = path.split(".")
        # Szukamy liniowo po wcieciach: config jest plaski (sekcja + klucze), wiec nie
        # potrzebujemy parsera YAML, a bez niego skrypt dziala takze tam, gdzie nie ma pyyaml.
        depth, i, hit = 0, 0, None
        while i < len(lines):
            m = re.match(r"^(\s*)([A-Za-z_][\w]*):(.*)$", lines[i])
            if m:
                ind, key, rest = len(m.group(1)), m.group(2), m.group(3).strip()
                if ind == depth * 2 and key == parts[depth]:
                    if depth == len(parts) - 1:
                        hit = (i, ind, rest)
                        break
                    depth += 1
            i += 1
        if hit is None:
            # Klucza nie ma, ale sekcja moze byc -- wtedy go DOPISUJEMY. Tak jest z `reg.space`,
            # ktore ma wartosc domyslna w kodzie i nie stoi w zadnym configu; bez tego sweep nie
            # moglby go w ogole dotknac.
            sec = _section_end(lines, parts[:-1])
            if sec is None:
                raise SystemExit(f"nie znalazlem sciezki {path!r} w {a.base}")
            j, ind = sec
            val = _typed(raw, None)
            val = str(val).lower() if isinstance(val, bool) else val
            lines.insert(j, f"{' ' * ind}{parts[-1]}: {val}")
            continue
        i, ind, old = hit
        val = _typed(raw, old or None)
        val = str(val).lower() if isinstance(val, bool) else val
        lines[i] = f"{' ' * ind}{parts[-1]}: {val}"

    txt = "\n".join(lines)
    base_out = re.search(r"^output_dir:\s*(\S+)", txt, re.M).group(1)
    txt = txt.replace(f"output_dir: {base_out}", f"output_dir: ./outputs/sweep/{name}", 1)
    txt = re.sub(r"^(\s*name:)\s*\S+\s*$", rf"\1 {name}", txt, count=1, flags=re.M)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print(f"[mkcfg] {a.out}")


if __name__ == "__main__":
    main()
