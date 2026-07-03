from __future__ import annotations

from pathlib import Path

from common import check_args, resolve_path, setup_log_file, write_json
from train_stage1 import parse_args, train_stage1


def main() -> None:
    args = parse_args()
    args.stage = 3
    check_args(args)
    Path(resolve_path(args.save_dir)).mkdir(parents=True, exist_ok=True)
    if not args.log_file:
        args.log_file = str(Path(args.save_dir) / f"stage3_c2shared_cvq_c36_snr{args.snr_db:g}_k{int(args.k)}.log")
    setup_log_file(args.log_file)
    write_json(Path(args.save_dir) / "stage3_args.json", vars(args))
    train_stage1(args)


if __name__ == "__main__":
    main()
