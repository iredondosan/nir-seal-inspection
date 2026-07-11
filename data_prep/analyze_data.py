#!/usr/bin/env python3
"""Audit: packs used to train the SEAL model and the DEFECT model per product, totals, and the split."""
import xml.etree.ElementTree as ET, glob, random
from collections import defaultdict
from seal_inspection.paths import ROOT
def load(x): return ET.parse(f"{ROOT}/{x}").getroot().findall("image")
def tg(im): return {t.get("label") for t in im.findall("tag")}
def nsell(im): return len([p for p in im.findall("polygon") if p.get("label")=="sellado"])
def ndef(im): return len([p for p in im.findall("polygon") if p.get("label")=="defect"])
IMG={"prod1":"prod1","prod2":"prod2","prod3":"prod3","prod4":"prod4","prod5":"prod5","prod6(good)":"prod6","prod6(bad)":"prod6_bad"}
def total(p): return len([f for f in glob.glob(f"{ROOT}/data/images/{IMG[p]}/*") if f.lower().endswith(('.png','.jpg'))])

# ---- SEAL: reviewed-filtered for prod1/prod2/prod6; all-sellado for prod3/4/5 ----
SEAL=[("prod1","data/annotations/prod1_reviewed.xml",True),("prod2","data/annotations/prod2_reviewed.xml",True),
      ("prod3","data/annotations/prod3.xml",False),("prod4","data/annotations/prod4.xml",False),
      ("prod5","data/annotations/prod5.xml",False),("prod6(good)","data/annotations/prod6_reviewed.xml",True),
      ("prod6(bad)","data/annotations/prod6_bad_reviewed.xml",True)]
print("===================== SEAL MODEL =====================")
print(f"{'product':11}{'total imgs':>11}{'seal-GT':>9}{'val':>5}{'train':>7}")
seal_tot=[0,0,0]
for prod,xml,rev in SEAL:
    try: imgs=load(xml)
    except Exception: print(f"{prod:11}  (missing {xml})"); continue
    packs=[im for im in imgs if "exclude" not in tg(im) and nsell(im)>=2 and (not rev or "reviewed" in tg(im))]
    n=len(packs); vp=2 if n>=6 else 0
    seal_tot[0]+=n; seal_tot[1]+=vp; seal_tot[2]+=n-vp
    print(f"{prod:11}{total(prod):>11}{n:>9}{vp:>5}{n-vp:>7}")
print(f"{'TOTAL':11}{'':>11}{seal_tot[0]:>9}{seal_tot[1]:>5}{seal_tot[2]:>7}")

# ---- DEFECT: prod1/prod2 'all'; prod6 good negatives; prod6 bad defect ----
DEFECT=[("prod1","data/annotations/prod1_reviewed.xml","all"),("prod2","data/annotations/prod2_reviewed.xml","all"),
        ("prod6(good)","data/annotations/prod6_reviewed.xml","good"),("prod6(bad)","data/annotations/prod6_bad_reviewed.xml","defect_reviewed")]
defect=[]; good=[]
for prod,xml,mode in DEFECT:
    try: imgs=load(xml)
    except Exception: continue
    for im in imgs:
        if "exclude" in tg(im) or nsell(im)<2: continue
        d=ndef(im); nm=im.get("name")
        if mode=="good":
            if tg(im)&{"good","reviewed"}: good.append((prod,nm))
        elif mode=="defect_reviewed":
            if "reviewed" in tg(im) and d>=1: defect.append((prod,nm))
        else:
            if tg(im)&{"good","defect","reviewed"}: (defect if d>=1 else good).append((prod,nm))
FT={"prod6_bad_003.jpg"}; FR={"prod6_bad_001.jpg","prod6_bad_002.jpg"}
rng=random.Random(42); rng.shuffle(defect); rng.shuffle(good)
dr=[x for x in defect if x[1] not in FT and x[1] not in FR]
dtest=set(id(x) for x in dr[:max(1,int(len(dr)*0.2))]); gtest=set(id(x) for x in good[:int(len(good)*0.2)])
res=defaultdict(lambda:[0,0,0,0])
for x in defect:
    prod,nm=x; test = nm in FT or (nm not in FR and id(x) in dtest); res[prod][1 if test else 0]+=1
for x in good:
    prod,nm=x; res[prod][3 if id(x) in gtest else 2]+=1
print("\n===================== DEFECT MODEL =====================")
print(f"{'product':11}{'total imgs':>11}{'def train':>10}{'def test':>9}{'good train':>11}{'good test':>10}")
tt=[0,0,0,0]
for prod in ["prod1","prod2","prod6(good)","prod6(bad)"]:
    r=res[prod];
    if sum(r)==0: continue
    for i in range(4): tt[i]+=r[i]
    print(f"{prod:11}{total(prod):>11}{r[0]:>10}{r[1]:>9}{r[2]:>11}{r[3]:>10}")
print(f"{'TOTAL':11}{'':>11}{tt[0]:>10}{tt[1]:>9}{tt[2]:>11}{tt[3]:>10}")
print(f"\nNOTE: ~80 unannotated prod6(bad) repeat scans excluded; defect model = {tt[0]+tt[1]} defect + {tt[2]+tt[3]} good labeled packs")
