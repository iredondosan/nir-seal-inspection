import torch, sys, collections
for m in ["models/best_lite_multiprod.pt", "models/best_lite_reviewed_1280.pt"]:
    print("=====", m, "=====")
    try:
        d = torch.load(m, map_location="cpu", weights_only=False)
    except Exception as e:
        print("  load error:", e); continue
    print("  type:", type(d).__name__)
    if isinstance(d, dict):
        meta = {k: v for k, v in d.items() if not hasattr(v, "shape") and k not in ("model", "state_dict")}
        subw = [k for k in d if k in ("model", "state_dict")]
        print("  non-tensor meta keys:", list(meta.keys()))
        for k, v in meta.items():
            s = str(v)
            print("    ", k, "=", s[:200])
        print("  weight containers:", subw)
        # param count
        sd = d.get("model", d.get("state_dict", d))
        if isinstance(sd, dict):
            n = sum(int(t.numel()) for t in sd.values() if hasattr(t, "numel"))
            print("  #params:", n)
    else:
        print("  (raw state_dict, no metadata)")
