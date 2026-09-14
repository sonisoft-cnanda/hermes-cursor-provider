"""Command-line entry point for the standalone Cursor provider."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .installer import (
    PLUGIN_INIT,
    PLUGIN_MANIFEST,
    _validate_token,
    install_plugin,
    uninstall_plugin,
)
from .runner import CursorRunner, RunnerConfig
from .server import BridgeApplication, create_http_server


def _default_hermes_home() -> str:
    return os.getenv("HERMES_HOME", "").strip() or str(Path.home() / ".hermes")


def is_supported_platform(platform_name: str | None = None) -> bool:
    """Native Windows lacks the required Cursor/process-tree guarantees."""
    return (platform_name or os.name) == "posix"


def _read_dotenv_value(path: Path, key: str) -> str:
    """Read one literal dotenv value without evaluating shell syntax."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeError):
        return ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if not line.startswith(f"{key}="):
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if any(character in value for character in "\r\n\x00"):
            return ""
        return value
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-cursor-provider",
        description="Standalone Cursor CLI model provider for Hermes Agent",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install", help="install the Hermes provider shim")
    install.add_argument("--hermes-home", default=_default_hermes_home())
    install.add_argument(
        "--force",
        action="store_true",
        help="back up and replace a conflicting Cursor provider",
    )

    uninstall = subparsers.add_parser("uninstall", help="remove the Hermes provider shim")
    uninstall.add_argument("--hermes-home", default=_default_hermes_home())
    uninstall.add_argument("--force", action="store_true", help="remove modified owned plugin files")

    serve = subparsers.add_parser("serve", help="run the authenticated loopback bridge")
    serve.add_argument("--hermes-home", default=_default_hermes_home())
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--cursor-command", default="cursor-agent")
    serve.add_argument("--cursor-arg", action="append", default=[], dest="cursor_args")
    serve.add_argument("--mode", choices=("hermes", "agent", "ask", "plan"), default="hermes")
    serve.add_argument("--workspace")
    serve.add_argument("--cursor-home")
    serve.add_argument("--timeout-seconds", type=float, default=1800.0)
    serve.add_argument("--max-concurrency", type=int, default=1)


    doctor = subparsers.add_parser("doctor", help="check installation and Cursor CLI visibility")
    doctor.add_argument("--hermes-home", default=_default_hermes_home())
    doctor.add_argument("--cursor-command", default="cursor-agent")
    doctor.add_argument("--cursor-home")

    return parser


def _serve(args: argparse.Namespace) -> int:
    if args.mode == "agent":
        print(
            "WARNING: agent mode enables Cursor-internal shell/write/edit operations; "
            "Hermes approvals do not gate them.",
            file=sys.stderr,
        )
    env_path = Path(args.hermes_home).expanduser() / ".env"
    token = (
        os.getenv("CURSOR_BRIDGE_API_KEY", "").strip()
        or _read_dotenv_value(env_path, "CURSOR_BRIDGE_API_KEY")
    )
    if not token:
        print(
            "error: CURSOR_BRIDGE_API_KEY is missing; run `hermes-cursor-provider install` "
            "and load the generated Hermes .env",
            file=sys.stderr,
        )
        return 2
    try:
        if not (0 <= args.port <= 65535):
            raise ValueError("port must be between 0 and 65535")
        config = RunnerConfig(
            command=args.cursor_command,
            extra_args=tuple(args.cursor_args),
            mode=args.mode,
            workspace=args.workspace,
            cursor_home=args.cursor_home or str(Path(args.hermes_home).expanduser() / "cursor-home"),
            timeout_seconds=args.timeout_seconds,
            cursor_api_key=(
                os.getenv("CURSOR_API_KEY", "").strip()
                or _read_dotenv_value(env_path, "CURSOR_API_KEY")
                or None
            ),
        )
        runner = CursorRunner(config)
        app = BridgeApplication(
            runner=runner,
            token=token,
            max_concurrency=args.max_concurrency,
        )
        server = create_http_server(
            host="127.0.0.1",
            port=args.port,
            app=app,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    bound_port = server.server_address[1]
    print(f"Cursor bridge listening on http://127.0.0.1:{bound_port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        runner.close()
    return 0


def _doctor(args: argparse.Namespace) -> int:
    home = Path(args.hermes_home).expanduser()
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    env_path = home / ".env"
    try:
        plugin_ok = (
            (plugin_dir / "__init__.py").read_text(encoding="utf-8") == PLUGIN_INIT
            and (plugin_dir / "plugin.yaml").read_text(encoding="utf-8") == PLUGIN_MANIFEST
        )
    except (FileNotFoundError, OSError, UnicodeError):
        plugin_ok = False
    try:
        token_value = _read_dotenv_value(env_path, "CURSOR_BRIDGE_API_KEY")
        token_ok = bool(_validate_token(token_value))
        token_mode_ok = (env_path.stat().st_mode & 0o777) == 0o600
    except (FileNotFoundError, OSError, ValueError):
        token_ok = False
        token_mode_ok = False
    cursor_path = shutil.which(args.cursor_command)
    print(f"plugin: {'ok' if plugin_ok else 'missing'}")
    print(f"bridge token: {'configured' if token_ok else 'missing'}")
    print(f"bridge token permissions: {'ok' if token_mode_ok else 'invalid'}")
    print(f"cursor-agent: {cursor_path or 'missing'}")
    cursor_ok = False
    if cursor_path:
        try:
            runner = CursorRunner(
                RunnerConfig(
                    command=args.cursor_command,
                    cursor_home=args.cursor_home or str(home / "cursor-home"),
                )
            )
            diagnostics = runner.diagnostics()
            print(f"cursor version: {diagnostics['version']}")
            print("cursor authentication: ok")
            print(f"cursor models: {diagnostics['model_count']}")
            runner.close()
            cursor_ok = True
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"cursor readiness: failed ({type(exc).__name__})")
    return 0 if plugin_ok and token_ok and token_mode_ok and cursor_ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command in {"install", "serve", "doctor"} and not is_supported_platform():
        print(
            "error: native Windows is unsupported; run the Cursor provider inside WSL instead",
            file=sys.stderr,
        )
        return 2

    if args.command == "install":
        try:
            result = install_plugin(args.hermes_home, force=args.force)
        except (FileExistsError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Installed Cursor provider: {result.plugin_dir}")
        if result.backup_dir is not None:
            print(f"Previous Cursor provider backup: {result.backup_dir}")
        print(f"Bridge credential stored in: {result.env_path}")
        return 0
    if args.command == "uninstall":
        try:
            uninstall_plugin(args.hermes_home, force=args.force)
        except (FileExistsError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("Removed Cursor provider plugin files; bridge credential was preserved.")
        return 0
    if args.command == "serve":
        return _serve(args)
    if args.command == "doctor":
        return _doctor(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
