"""mapping_first_frame の共有メモリがspawnプロセス間で動作するか検証"""
import torch
import torch.multiprocessing as mp
import time, sys

def writer(tensor):
    print(f"[Writer] PID started, tensor value = {tensor[0].item()}", flush=True)
    time.sleep(2)
    tensor[0] = 1
    print(f"[Writer] Set tensor to {tensor[0].item()}", flush=True)

def reader(tensor):
    print(f"[Reader] PID started, tensor value = {tensor[0].item()}", flush=True)
    for i in range(10):
        val = tensor[0].item()
        print(f"[Reader] Check {i}: value = {val}", flush=True)
        if val == 1:
            print("[Reader] SUCCESS: saw updated value!", flush=True)
            return
        time.sleep(1)
    print("[Reader] FAILURE: never saw updated value", flush=True)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    t = torch.zeros((1), device='cpu').int()
    t.share_memory_()
    print(f"[Main] Created shared tensor: {t[0].item()}, is_shared={t.is_shared()}", flush=True)

    p1 = mp.Process(target=reader, args=(t,))
    p2 = mp.Process(target=writer, args=(t,))
    p1.start(); p2.start()
    p1.join(); p2.join()
    print(f"[Main] Final value: {t[0].item()}", flush=True)
