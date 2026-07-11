#!/usr/bin/env python3
"""Per-product breakdown of labelled packs by reviewed-status x defect/good, to size the global hold-out."""
import os, numpy as np, cv2
import xml.etree.ElementTree as ET
from seal_inspection.paths import ROOT
SRC=[("prod1","data/annotations/prod1_reviewed.xml"),("prod2","data/annotations/prod2_reviewed.xml"),
     ("prod3","data/annotations/prod3_reviewed.xml"),("prod4","data/annotations/prod4_reviewed.xml"),
     ("prod5","data/annotations/prod5_reviewed.xml"),("prod6","data/annotations/prod6_reviewed.xml")]
def tags(n): return {t.get("label") for t in n.findall("tag")}
def has_seal(n): return len([p for p in n.findall("polygon") if p.get("label")=="sellado"])>=2
def is_def(n): return any(p.get("label") in ("defect","liquid") for p in n.findall("polygon"))
hdr=f"{'product':9} {'packs':>6} {'reviewed':>9} | {'def':>4} {'def/rev':>7} {'def/NOTrev':>10} | {'good':>5} {'good/rev':>8} {'good/NOTrev':>11}"
print(hdr); print("-"*len(hdr))
tot=np.zeros(7,int)
for prod,xml in SRC:
    p=f"{ROOT}/{xml}"
    if not os.path.exists(p): continue
    packs=rev=d=dr=dn=g=gr=gn=0
    for n in ET.parse(p).getroot().findall("image"):
        if "exclude" in tags(n): continue
        if not has_seal(n): continue
        packs+=1; r="reviewed" in tags(n); rev+=r
        if is_def(n):
            d+=1; dr+=r; dn+=(not r)
        else:
            g+=1; gr+=r; gn+=(not r)
    print(f"{prod:9} {packs:6} {rev:9} | {d:4} {dr:7} {dn:10} | {g:5} {gr:8} {gn:11}")
    tot+=np.array([packs,rev,d,dr,dn,g,gr,gn][:7])  # packs,rev,d,dr,dn,g,gr ... fix below
