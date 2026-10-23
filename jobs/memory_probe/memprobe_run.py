# -*- coding: utf-8 -*-
"""Peak GPU memory probe. Runs an existing script unchanged in this process and prints the
PyTorch peak memory counters when it finishes. No file of the repository is modified.

usage:  python memprobe_run.py <script.py> <the script's own arguments ...>
prints: [memprobe] peak allocated X.XX GiB, peak reserved Y.YY GiB (<GPU name>, total Z GiB)
"""
import sys, os, runpy, atexit

if len(sys.argv) < 2:
    sys.exit("usage: python memprobe_run.py <script.py> [args...]")
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]          # the probed script sees its own argv
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))

def report():
    try:
        import torch
        if not torch.cuda.is_available():
            print("[memprobe] no CUDA device", flush=True); return
        torch.cuda.synchronize()
        dev = torch.cuda.current_device()
        alloc = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        resv = torch.cuda.max_memory_reserved(dev) / 2 ** 30
        total = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
        print(f"[memprobe] peak allocated {alloc:.2f} GiB, peak reserved {resv:.2f} GiB "
              f"({torch.cuda.get_device_name(dev)}, total {total:.1f} GiB)", flush=True)
    except Exception as e:  # never hide the probed script's own result
        print(f"[memprobe] report failed: {e!r}", flush=True)

atexit.register(report)
runpy.run_path(script, run_name="__main__")
