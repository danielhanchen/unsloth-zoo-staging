import json, sys, statistics as st, collections
rows=[json.loads(l) for l in open(sys.argv[1])]
g=collections.defaultdict(list)
for r in rows: g[(r["model"], r["zoo"], r["arm"])].append(r)
texts={}
for k,v in sorted(g.items()):
    t=[x["gen_tps"] for x in v]
    print(k, "n=%d"%len(t), "median tok/s %.2f"%st.median(t), "spread %.2f-%.2f"%(min(t),max(t)), "calls", sorted({x["calls"] for x in v}), "patched", v[0]["patched"], "identical_texts", len({x["text"] for x in v})==1)
    texts[k]=v[0]["text"]
for m in {k[0] for k in texts}:
    ref=[t for k,t in texts.items() if k[0]==m and k[2]=="native"][0]
    for k,t in texts.items():
        if k[0]==m: print(m, k[1], k[2], "text==native:", t==ref)
