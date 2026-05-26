import argparse
import shutil
from pathlib import Path


DEFAULT_REPO_ID = "BeichenZhang/LongCLIP-L"
DEFAULT_FILENAME = "longclip-L.pt"
DEFAULT_OUTPUT = "LongCLIP/checkpoints/longclip-L.pt"
MIN_EXPECTED_BYTES = 1_000_000_000


def parse_args():
    parser = argparse.ArgumentParser(description="Download the LongCLIP checkpoint used by CLIP-T.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repository id.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Checkpoint filename in the Hugging Face repository.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Local checkpoint path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        size = output_path.stat().st_size
        if size >= MIN_EXPECTED_BYTES:
            print(f"LongCLIP checkpoint already exists: {output_path}")
            return 0
        print(f"Existing checkpoint looks incomplete ({size} bytes): {output_path}")
        print("Run again with --force to replace it.")
        return 1

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub with `pip install -r requirements.txt`.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(output_path.parent),
        )
    )

    if downloaded.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        shutil.copy2(downloaded, output_path)

    size = output_path.stat().st_size
    if size < MIN_EXPECTED_BYTES:
        raise SystemExit(f"Downloaded checkpoint looks incomplete ({size} bytes): {output_path}")

    print(f"LongCLIP checkpoint ready: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
