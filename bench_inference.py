"""
bench_inference.py — inference의 elapsed time + peak VRAM 측정
run_memo.sh의 각 inference 명령을 감싸서 벤치마크 수행.

Usage (shell에서 호출):
  python bench_inference.py --method DDIM --output_csv bench.csv -- python -m examples.text_to_mscoco ...

CSV output:
  method, num_samples, total_time_sec, per_sample_time_ms, peak_vram_MB
"""
import argparse
import subprocess
import sys
import os
import csv
import time
import threading


def poll_vram(gpu_id, interval, stop_event, max_vram):
    """백그라운드에서 GPU VRAM 폴링 (peak 추적)"""
    try:
        import torch
        while not stop_event.is_set():
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024 * 1024)
            peak = max(allocated, reserved)
            max_vram[0] = max(max_vram[0], peak)
            stop_event.wait(interval)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference: time + VRAM")
    parser.add_argument("--method", required=True, help="method name (DDIM, CNO, init_opti)")
    parser.add_argument("--num_samples", type=int, required=True, help="number of images generated")
    parser.add_argument("--output_csv", required=True, help="output CSV path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="inference command (after --)")
    args = parser.parse_args()

    # Remove leading "--" if present
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[ERROR] No command provided after --")
        sys.exit(1)

    # nvidia-smi 백그라운드 폴러 (subprocess VRAM 추적)
    vram_samples = []
    stop_event = threading.Event()

    def smi_poller():
        import subprocess as sp
        while not stop_event.is_set():
            try:
                result = sp.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                     "--id", str(args.gpu)],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    vram_mb = float(result.stdout.strip())
                    vram_samples.append(vram_mb)
            except Exception:
                pass
            stop_event.wait(0.1)  # 100ms interval

    print(f"[BENCH] method={args.method} num_samples={args.num_samples}")
    print(f"[BENCH] command: {' '.join(cmd[:3])}...")

    # VRAM 폴러 시작
    poller_thread = threading.Thread(target=smi_poller, daemon=True)
    poller_thread.start()

    # inference 실행 + 시간 측정
    t_start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=os.getcwd())
    t_end = time.perf_counter()

    # VRAM 폴러 정지
    stop_event.set()
    poller_thread.join(timeout=2)

    total_time = t_end - t_start
    per_sample_ms = (total_time / args.num_samples) * 1000 if args.num_samples > 0 else 0
    peak_vram_mb = max(vram_samples) if vram_samples else -1
    peak_vram_gb = peak_vram_mb / 1024 if peak_vram_mb > 0 else -1

    # CSV 저장
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    file_exists = os.path.isfile(args.output_csv)

    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["method", "num_samples", "total_time_sec",
                             "per_sample_time_ms", "peak_vram_GB"])
        writer.writerow([args.method, args.num_samples,
                         f"{total_time:.2f}", f"{per_sample_ms:.1f}", f"{peak_vram_gb:.2f}"])

    print(f"[BENCH] {args.method}: total={total_time:.2f}s  "
          f"per_sample={per_sample_ms:.1f}ms  peak_VRAM={peak_vram_gb:.2f}GB")
    print(f"[BENCH] saved → {args.output_csv}")

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
