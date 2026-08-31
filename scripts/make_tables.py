"""Tabele artykulu z cifc_metrics.json: per koncept (uklad Tabel 1-2 CIDM) + zbiorcza liga.
Zero GPU. Markdown + LaTeX do assets/tables/. Weryfikacja: srednia per koncept == average_final."""
import argparse, json, os, statistics as st

CON = ["V1 dog","V2 duck","V3 cat","V4 backp.","V5 teddy","V6 paint.","V7 dog2","V8 draw.","V9 cat2","V10 ink"]

def per_concept(path):
    d = json.load(open(path)); m = d["matrix"]
    K = max(int(k.split(",")[0]) for k in m)
    rows = [m[f"{K},{j}"] for j in range(K + 1)]
    for metric in ("clip_t", "clip_i", "dino_i"):
        avg = st.mean(r[metric] for r in rows)
        assert abs(avg - d["average_final"][metric]) < 1e-6, f"weryfikacja {metric}: {avg} != {d['average_final'][metric]}"
    return rows, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True, help="cifc_metrics.json naszych seedow (1+)")
    ap.add_argument("--league", nargs="*", default=[], help="etykieta=sciezka dla ligi")
    ap.add_argument("--out", default="assets/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    seeds = [per_concept(p) for p in a.ours]
    md = ["| koncept | DINO | TA | IA |", "|---|---|---|---|"]
    tex = ["\\begin{tabular}{lccc}", "\\toprule", "koncept & DINO & TA & IA \\\\", "\\midrule"]
    for j, name in enumerate(CON):
        di = [s[0][j]["dino_i"] for s in seeds]; ta = [s[0][j]["clip_t"] for s in seeds]
        ia = [s[0][j]["clip_i"] for s in seeds]
        f = lambda v: (f"{st.mean(v):.3f}±{st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
        md.append(f"| {name} | {f(di)} | {f(ta)} | {f(ia)} |")
        tex.append(f"{name} & {f(di)} & {f(ta)} & {f(ia)} \\\\")
    avg = lambda k: [s[1]["average_final"][k] for s in seeds]
    f = lambda v: (f"{st.mean(v):.3f}±{st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
    md.append(f"| **średnia** | **{f(avg('dino_i'))}** | **{f(avg('clip_t'))}** | **{f(avg('clip_i'))}** |")
    tex += ["\\midrule", f"śr. & {f(avg('dino_i'))} & {f(avg('clip_t'))} & {f(avg('clip_i'))} \\\\",
            "\\bottomrule", "\\end{tabular}"]
    open(f"{a.out}/per_concept.md", "w").write("\n".join(md) + "\n")
    open(f"{a.out}/per_concept.tex", "w").write("\n".join(tex) + "\n")

    if a.league:
        md = ["| metoda | TA | IA | DINO | Fgt(DINO) |", "|---|---|---|---|---|"]
        tex = ["\\begin{tabular}{lcccc}", "\\toprule", "metoda & TA & IA & DINO & Fgt \\\\", "\\midrule"]
        rows = [("ContinualHyper", None)] + [tuple(x.split("=", 1)) for x in a.league]
        for lab, p in rows:
            if p is None:
                v = {k: f(avg(k)) for k in ("clip_t", "clip_i", "dino_i")}
                fg = f(list([s[1]["forgetting"]["dino_i"] for s in seeds]))
            else:
                d = json.load(open(p)); af = d["average_final"]
                v = {k: f"{af[k]:.3f}" for k in af}
                fg = f"{d['forgetting']['dino_i']:.3f}"
            md.append(f"| {lab} | {v['clip_t']} | {v['clip_i']} | {v['dino_i']} | {fg} |")
            tex.append(f"{lab} & {v['clip_t']} & {v['clip_i']} & {v['dino_i']} & {fg} \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        open(f"{a.out}/league.md", "w").write("\n".join(md) + "\n")
        open(f"{a.out}/league.tex", "w").write("\n".join(tex) + "\n")
    print("zapisane do", a.out)

main()
