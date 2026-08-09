from __future__ import annotations

from pathlib import Path

from hermes_cursor_provider import cli
from hermes_cursor_provider.cli import build_parser, main


def test_serve_defaults_to_read_only_ask_mode() -> None:
    args = build_parser().parse_args(["serve"])
    assert args.mode == "ask"


def test_agent_mode_warns_that_hermes_approvals_do_not_apply(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        ["serve", "--mode", "agent", "--hermes-home", str(tmp_path / "hermes")]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Hermes approvals do not gate" in captured.err


def test_install_fails_closed_on_unsupported_native_windows(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    home = tmp_path / "hermes"
    monkeypatch.setattr(cli, "is_supported_platform", lambda: False, raising=False)

    exit_code = main(["install", "--hermes-home", str(home)])

    assert exit_code == 2
    assert "native windows" in capsys.readouterr().err.lower()
    assert not (home / "plugins").exists()


def test_install_command_does_not_print_generated_token(tmp_path: Path, capsys) -> None:
    home = tmp_path / "hermes"

    exit_code = main(["install", "--hermes-home", str(home)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Installed Cursor provider" in captured.out
    token = next(
        line.split("=", 1)[1]
        for line in (home / ".env").read_text().splitlines()
        if line.startswith("CURSOR_BRIDGE_API_KEY=")
    )
    assert token
    assert token not in captured.out
    assert token not in captured.err


def test_uninstall_command_removes_provider(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    assert main(["install", "--hermes-home", str(home)]) == 0

    exit_code = main(["uninstall", "--hermes-home", str(home)])

    assert exit_code == 0
    assert not (home / "plugins" / "model-providers" / "cursor").exists()


def test_serve_requires_bridge_token(monkeypatch, capsys) -> None:
    monkeypatch.delenv("CURSOR_BRIDGE_API_KEY", raising=False)

    exit_code = main(["serve", "--port", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "CURSOR_BRIDGE_API_KEY" in captured.err


def test_serve_does_not_offer_a_bind_override(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CURSOR_BRIDGE_API_KEY", "local-secret")

    exit_code = main(["serve", "--host", "0.0.0.0", "--port", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unrecognized arguments" in captured.err.lower()


def test_serve_reads_token_from_selected_hermes_env(tmp_path: Path, monkeypatch, capsys) -> None:
    home = tmp_path / "hermes"
    assert main(["install", "--hermes-home", str(home)]) == 0
    monkeypatch.delenv("CURSOR_BRIDGE_API_KEY", raising=False)

    exit_code = main(
        [
            "serve",
            "--hermes-home",
            str(home),
            "--port",
            "-1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "port must be between" in captured.err.lower()
    assert "CURSOR_BRIDGE_API_KEY is missing" not in captured.err


def test_help_does_not_offer_token_or_cursor_key_flags(capsys) -> None:
    exit_code = main(["serve", "--help"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "--token" not in captured.out
    assert "--cursor-api-key" not in captured.out
    assert "--host" not in captured.out
    assert "--unsafe-allow-non-loopback" not in captured.out
