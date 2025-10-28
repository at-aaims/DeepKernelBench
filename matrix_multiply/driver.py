from __future__ import annotations
import argparse, importlib
from pathlib import Path
import pandas as pd
import concurrent.futures as cf
import torch

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterate through a CSV of parameters and launch benchmarks."
    )
    p.add_argument(
        "--bench-script",
        default="bench",
        help="Python module that exposes `run_benchmark(**kwargs)` (default: bench)",
    )
    p.add_argument(
        "--csv-file",
        type=Path,
        default=Path("params.csv"),
        help="CSV describing the sweep grid (default: params.csv)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Number of parallel processes (0 = run serially, default: 0)",
    )
    p.add_argument("--save", action="store_true", help="Save the results in CSV")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    bench = importlib.import_module(args.bench_script)
    df = pd.read_csv(args.csv_file, comment='#')

    def _call(row):
        idx, params = row
        print(f"[{idx}] running with {params}")
        metrics = bench.run_benchmark(**params)
        params['Performance (TFLOPS)'] = metrics
        return idx, params  # 2 elements are required for dict()

    iterable = df.to_dict(orient="index").items()
    if args.max_workers:
        with cf.ProcessPoolExecutor(args.max_workers) as pool:
            results = dict(pool.map(_call, iterable))
    else:
        results = dict(map(_call, iterable))

    if args.save:
        device_name = torch.cuda.get_device_name(0).replace(' ', '_')
        input_csv = str(args.csv_file).replace('/', '_')
        df = pd.DataFrame.from_records(tuple(results.values()))
        save_file = f"{args.bench_script}_{device_name}_{input_csv}"
        df.to_csv(save_file)
        print(f"Saved results to {save_file}")

if __name__ == "__main__":
    main()

