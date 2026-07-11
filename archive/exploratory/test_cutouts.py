#!/usr/bin/env python3
"""Standalone check: how many defect/liquid cut-outs does the new all-packs gatherer pull?"""
import os, numpy as np, cv2
import xml.etree.ElementTree as ET
from seal_inspection.paths import ROOT
DATASETS=[("data/annotations/prod2_reviewed.xml","data/images/prod2","prod2"),
          ("data/annotations/prod1_reviewed.xml","data/images/prod1","prod1"),
          ("data/annotations/prod3_reviewed.xml","data/images/prod3","prod3"),
          ("data/annotations/prod4_reviewed.xml","data/images/prod4","prod4"),
          ("data/annotations/prod5_reviewed.xml","data/images/prod5","prod5"),
          ("data/annotations/prod6_reviewed.xml","data/images/prod6","prod6"),
          ("data/annotations/prod6_bad_reviewed.xml","data/images/prod6_bad","prod6")]
CONTAM_EXCLUDE={"seal_1302_1780665903828_raw.png"}
def parse_pts(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
insts=[]; per={}
for xmlrel,imgrel,prod in DATASETS:
    xmlp=f"{ROOT}/{xmlrel}"; c=0
    if not os.path.exists(xmlp): continue
    for node in ET.parse(xmlp).getroot().findall("image"):
        name=node.get("name")
        if name in CONTAM_EXCLUDE: continue
        polys=[pg for pg in node.findall("polygon") if pg.get("label") in ("defect","liquid")]
        if not polys: continue
        p=f"{ROOT}/{imgrel}/{name}"
        if not os.path.exists(p): continue
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        for pg in polys:
            pts=parse_pts(pg.get("points")).astype(np.int32); x,y,bw,bh=cv2.boundingRect(pts)
            if bw<4 or bh<4: continue
            insts.append((g[y:y+bh,x:x+bw].copy(),)); c+=1
    per[prod]=per.get(prod,0)+c
print("per-source cut-outs:",per)
print("TOTAL defect/liquid cut-outs from all labelled packs:",len(insts),"(was 26 from contaminants.xml)")
