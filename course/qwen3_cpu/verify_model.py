import argparse
import hashlib
import json
import sys
from pathlib import Path


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_usage(self):
        return super().format_usage().replace("usage:", "用法：", 1)

    def format_help(self):
        return super().format_help().replace("usage:", "用法：", 1)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 参数错误，请使用 --help 查看用法。\n")


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取模型清单：{path}") from exc
    except ValueError as exc:
        raise ValueError(f"模型清单不是有效的 JSON：{path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("模型清单必须是 JSON 对象")
    return manifest


def verify_model(model_path: Path, manifest: dict) -> None:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise ValueError(f"模型文件不存在：{model_path}")

    expected_size = manifest["size_bytes"]
    try:
        actual_size = model_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"无法读取模型文件：{model_path}") from exc
    if actual_size != expected_size:
        raise ValueError(f"模型文件大小不匹配：预期 {expected_size} 字节，实际 {actual_size} 字节")

    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"无法读取模型文件：{model_path}") from exc

    actual_sha256 = digest.hexdigest()
    if actual_sha256.lower() != manifest["sha256"].lower():
        raise ValueError(f"SHA-256 不匹配：预期 {manifest['sha256']}，实际 {actual_sha256}")


def main(argv=None) -> int:
    parser = ChineseArgumentParser(description="根据模型清单校验 GGUF 模型", add_help=False)
    options = parser.add_argument_group("选项")
    options.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    options.add_argument("--manifest", required=True, type=Path, metavar="路径", help="模型清单 JSON 文件路径")
    options.add_argument("--model", required=True, type=Path, metavar="路径", help="模型文件路径")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        verify_model(args.model, manifest)
    except (KeyError, ValueError, OSError) as exc:
        print(f"模型校验失败：{exc}", file=sys.stderr)
        return 1

    print(f"模型校验通过：{args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
