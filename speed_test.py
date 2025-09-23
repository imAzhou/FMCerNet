import os
import glob
import random
import time
from PIL import Image
from tqdm import tqdm

# 配置参数
IMG_DIR = "/nfs-medical3/zly/WS1600"  # NFS 挂载的图片目录
SAMPLE_SIZE = 10000          # 抽样数量，比如 10000 张
READ_MODE = "random"         # "sequential" 或 "random"

def seconds_to_hms(seconds: float) -> str:
    """把秒数转成 h:m:s 格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"

def benchmark(files):
    start = time.time()
    with tqdm(total=len(files), unit="img") as pbar:
        for f in files:
            try:
                with Image.open(f) as img:
                    img.load()  # 实际读取图像内容
            except Exception as e:
                tqdm.write(f"Error reading {f}: {e}")
            pbar.update(1)
            # 实时速度 (images/s)
            elapsed = time.time() - start
            if elapsed > 0:
                pbar.set_postfix(speed=f"{pbar.n/elapsed:.2f} img/s")
    end = time.time()
    total_time = end - start
    return total_time

if __name__ == "__main__":
    all_files = glob.glob(f"{IMG_DIR}/**/*.png")
    print(f"Total images found: {len(all_files)}")

    if READ_MODE == "random":
        sample_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    else:
        sample_files = all_files[:SAMPLE_SIZE]

    print(f"Testing with {len(sample_files)} images...")
    elapsed = benchmark(sample_files)

    avg_time_per_img = elapsed / len(sample_files)
    throughput = len(sample_files) / elapsed

    print(f"\n--- Benchmark Result ---")
    print(f"Total time: {elapsed:.2f} s")
    print(f"Avg time per image: {avg_time_per_img:.4f} s")
    print(f"Throughput: {throughput:.2f} images/s")

    # 推理全量耗时
    est_total_time = avg_time_per_img * len(all_files)
    print(f"\n--- Estimated Full Run ({len(all_files)} images) ---")
    print(f"Estimated time: {seconds_to_hms(est_total_time)}")
    