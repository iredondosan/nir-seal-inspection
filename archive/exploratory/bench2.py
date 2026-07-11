import torch, time, os, platform
from predict import U

def bench(dev, sizes, iters, batch=1):
    m=U().to(dev).eval()
    for (H,W) in sizes:
        x=torch.randn(batch,3,H,W,device=dev)
        with torch.no_grad():
            for _ in range(8): m(x)
        if dev=="cuda": torch.cuda.synchronize()
        t=time.time()
        with torch.no_grad():
            for _ in range(iters): m(x)
        if dev=="cuda": torch.cuda.synchronize()
        dt=(time.time()-t)/iters/batch*1000
        px=H*W
        print(f"  {dev}  {W}x{H} (b{batch}): {dt:7.2f} ms/img   {1000/dt:6.0f} img/s   [{px/1000:.0f}k px]")

m=U()
print("params: %.1f M"%(sum(p.numel() for p in m.parameters())/1e6))
try:
    with open("/proc/cpuinfo") as f:
        cpu=[l.split(":")[1].strip() for l in f if l.startswith("model name")][:1]
    print("CPU:", cpu[0] if cpu else "?", "| cores", os.cpu_count(), "| torch threads", torch.get_num_threads())
except Exception: print("CPU:", platform.processor(), os.cpu_count())
print("=== GPU (CUDA) ===")
bench("cuda",[(640,512),(384,384),(256,256)],iters=80)
print("=== GPU batched ===")
bench("cuda",[(384,384)],iters=40,batch=8)
print("=== CPU (proxy for embedded; this is a desktop CPU, Pi5 will be much slower) ===")
torch.set_num_threads(os.cpu_count())
bench("cpu",[(640,512),(384,384),(256,256)],iters=8)
