"""SNI_SLAMと同様のパターン: bound methodとしてspawnに渡す場合の検証"""
import torch
import torch.multiprocessing as mp
import time

class FakeSLAM:
    def __init__(self):
        self.mapping_first_frame = torch.zeros((1), device='cpu').int()
        self.mapping_first_frame.share_memory_()
        print(f"[Init] is_shared={self.mapping_first_frame.is_shared()}", flush=True)

    def tracking(self, rank):
        print(f"[Tracking] Started, is_shared={self.mapping_first_frame.is_shared()}, val={self.mapping_first_frame[0].item()}", flush=True)
        for i in range(15):
            val = self.mapping_first_frame[0].item()
            print(f"[Tracking] Check {i}: value = {val}", flush=True)
            if val == 1:
                print("[Tracking] SUCCESS", flush=True)
                return
            time.sleep(1)
        print("[Tracking] FAILURE: timeout", flush=True)

    def mapping(self, rank):
        print(f"[Mapping] Started, is_shared={self.mapping_first_frame.is_shared()}, val={self.mapping_first_frame[0].item()}", flush=True)
        time.sleep(3)
        self.mapping_first_frame[0] = 1
        print(f"[Mapping] Set to 1", flush=True)

    def run(self):
        mp.set_start_method('spawn', force=True)
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank,))
            else:
                p = mp.Process(target=self.mapping, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == '__main__':
    slam = FakeSLAM()
    slam.run()
