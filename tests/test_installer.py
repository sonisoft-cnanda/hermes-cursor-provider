from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_cursor_provider.installer as installer_module
from hermes_cursor_provider.installer import (
    PLUGIN_INIT,
    PLUGIN_MANIFEST,
    install_plugin,
    uninstall_plugin,
)

TEST_TOKEN = "bridge-test-token-" + "b" * 32
FIRST_TOKEN = "first-test-token-" + "f" * 32
SECOND_TOKEN = "second-test-token-" + "s" * 32


def test_checked_in_plugin_matches_installer_template() -> None:
    root = Path(__file__).resolve().parents[1]
    plugin = root / "plugin" / "model-providers" / "cursor"
    assert (plugin / "__init__.py").read_text() == PLUGIN_INIT
    assert (plugin / "plugin.yaml").read_text() == PLUGIN_MANIFEST


def test_install_rejects_weak_explicit_token_before_writing(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    with pytest.raises(ValueError, match="43"):
        install_plugin(home, token="too-short")
    assert not home.exists()

    with pytest.raises(ValueError, match="weak"):
        install_plugin(home, token="a" * 43)
    assert not home.exists()


def test_invalid_existing_bridge_token_does_not_leave_partial_plugin(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("CURSOR_BRIDGE_API_KEY=invalid\n")

    with pytest.raises(ValueError, match="43"):
        install_plugin(home, token=TEST_TOKEN)

    assert not (home / "plugins" / "model-providers" / "cursor").exists()


def test_install_writes_provider_plugin_and_preserves_existing_env(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    env_path = home / ".env"
    env_path.write_text("NOUS_API_KEY=keep-me\n")

    result = install_plugin(home, token=TEST_TOKEN)

    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    assert result.plugin_dir == plugin_dir
    assert (plugin_dir / "__init__.py").exists()
    assert (plugin_dir / "plugin.yaml").read_text().startswith("name: cursor\n")
    shim = (plugin_dir / "__init__.py").read_text()
    assert "ProviderProfile" in shim
    assert "hermes_cursor_provider" not in shim
    assert TEST_TOKEN not in shim
    env = env_path.read_text()
    assert "NOUS_API_KEY=keep-me" in env
    assert f"CURSOR_BRIDGE_API_KEY={TEST_TOKEN}" in env
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_install_is_idempotent_and_does_not_replace_existing_token(tmp_path: Path) -> None:
    home = tmp_path / "hermes"

    first = install_plugin(home, token=FIRST_TOKEN)
    second = install_plugin(home, token=SECOND_TOKEN)

    env = (home / ".env").read_text()
    assert env.count("CURSOR_BRIDGE_API_KEY=") == 1
    assert f"CURSOR_BRIDGE_API_KEY={FIRST_TOKEN}" in env
    assert first.token == second.token == FIRST_TOKEN


def test_install_refuses_to_overwrite_unmanaged_cursor_plugin(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    plugin_dir.mkdir(parents=True)
    custom = "# user-managed cursor provider\n"
    (plugin_dir / "__init__.py").write_text(custom)

    with pytest.raises(FileExistsError, match="force"):
        install_plugin(home, token=TEST_TOKEN)

    assert (plugin_dir / "__init__.py").read_text() == custom
    assert not (home / ".env").exists()


def test_force_install_backs_up_conflicting_provider_before_replacement(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("# stale\n")
    (plugin_dir / "custom.txt").write_text("preserve me\n")

    result = install_plugin(home, token=TEST_TOKEN, force=True)

    assert (plugin_dir / "__init__.py").read_text() == PLUGIN_INIT
    assert result.backup_dir is not None
    assert (result.backup_dir / "__init__.py").read_text() == "# stale\n"
    assert (result.backup_dir / "custom.txt").read_text() == "preserve me\n"


@pytest.mark.parametrize("symlink_component", ["plugins", "model-providers", "cursor"])
def test_install_rejects_symlinked_managed_provider_path(
    tmp_path: Path,
    symlink_component: str,
) -> None:
    home = tmp_path / "hermes"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside\n")

    if symlink_component == "plugins":
        home.mkdir()
        (home / "plugins").symlink_to(outside, target_is_directory=True)
    elif symlink_component == "model-providers":
        (home / "plugins").mkdir(parents=True)
        (home / "plugins" / "model-providers").symlink_to(outside, target_is_directory=True)
    else:
        parent = home / "plugins" / "model-providers"
        parent.mkdir(parents=True)
        (parent / "cursor").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        install_plugin(home, token=TEST_TOKEN, force=True)

    assert sentinel.read_text() == "outside\n"
    assert not (home / ".env").exists()


@pytest.mark.parametrize("leaf", [".env", "__init__.py", "plugin.yaml", "__pycache__"])
def test_install_rejects_symlinked_managed_leaf(
    tmp_path: Path,
    leaf: str,
) -> None:
    home = tmp_path / "hermes"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "target"
    target.write_text("sentinel\n")
    if leaf == ".env":
        home.mkdir()
        (home / leaf).symlink_to(target)
    else:
        plugin_dir = home / "plugins" / "model-providers" / "cursor"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / leaf).symlink_to(target, target_is_directory=leaf == "__pycache__")

    with pytest.raises(ValueError, match="symlink"):
        install_plugin(home, token=TEST_TOKEN, force=True)

    assert target.read_text() == "sentinel\n"


def test_install_rolls_back_new_env_when_plugin_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes"
    real_atomic_write = installer_module._atomic_write

    def fail_plugin_write(path: Path, content: str, *, mode: int | None = None) -> None:
        if path.name == "__init__.py":
            raise OSError("injected plugin write failure")
        real_atomic_write(path, content, mode=mode)

    monkeypatch.setattr(installer_module, "_atomic_write", fail_plugin_write)

    with pytest.raises(OSError, match="injected"):
        install_plugin(home, token=TEST_TOKEN)

    assert not (home / ".env").exists()
    assert not (home / "plugins" / "model-providers" / "cursor").exists()


def test_uninstall_rejects_symlinked_provider_directory_even_with_force(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_init = outside / "__init__.py"
    outside_manifest = outside / "plugin.yaml"
    outside_init.write_text(PLUGIN_INIT)
    outside_manifest.write_text(PLUGIN_MANIFEST)
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    plugin_dir.parent.mkdir(parents=True)
    plugin_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        uninstall_plugin(home, force=True)

    assert outside_init.read_text() == PLUGIN_INIT
    assert outside_manifest.read_text() == PLUGIN_MANIFEST


def test_uninstall_removes_only_owned_plugin_files(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    install_plugin(home, token=TEST_TOKEN)
    keep = home / "plugins" / "model-providers" / "other" / "keep.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("keep")

    uninstall_plugin(home)

    assert not (home / "plugins" / "model-providers" / "cursor").exists()
    assert keep.read_text() == "keep"
    assert "NOUS" not in (home / ".env").read_text()


def test_uninstall_preserves_modified_owned_file_without_force(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    install_plugin(home, token=TEST_TOKEN)
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    modified = "# local customization\n"
    (plugin_dir / "__init__.py").write_text(modified)

    with pytest.raises(FileExistsError, match="force"):
        uninstall_plugin(home)

    assert (plugin_dir / "__init__.py").read_text() == modified
    assert (plugin_dir / "plugin.yaml").exists()


def test_force_uninstall_removes_only_named_owned_files(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    install_plugin(home, token=TEST_TOKEN)
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    (plugin_dir / "__init__.py").write_text("# local customization\n")
    keep = plugin_dir / "keep.txt"
    keep.write_text("keep")

    uninstall_plugin(home, force=True)

    assert not (plugin_dir / "__init__.py").exists()
    assert not (plugin_dir / "plugin.yaml").exists()
    assert keep.read_text() == "keep"


def test_installed_profile_uses_standard_http_api_key_transport(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    install_plugin(home, token=TEST_TOKEN)
    shim = (home / "plugins" / "model-providers" / "cursor" / "__init__.py").read_text()

    assert "external_process" not in shim
    assert "cursor://" not in shim


def test_installed_shim_is_independent_of_bridge_python_environment(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    install_plugin(home, token=TEST_TOKEN)
    fake_core = tmp_path / "fake-core"
    providers = fake_core / "providers"
    providers.mkdir(parents=True)
    (providers / "__init__.py").write_text(
        "REGISTRY = {}\n"
        "def register_provider(profile): REGISTRY[profile.name] = profile\n"
    )
    (providers / "base.py").write_text(
        "class ProviderProfile:\n"
        "    def __init__(self, **kwargs): self.__dict__.update(kwargs)\n"
    )
    shim_path = home / "plugins" / "model-providers" / "cursor" / "__init__.py"
    script = "\n".join(
        [
            "import importlib.util, json, pathlib, sys",
            f"sys.path.insert(0, {str(fake_core)!r})",
            "import providers",
            f"path = pathlib.Path({str(shim_path)!r})",
            "spec = importlib.util.spec_from_file_location('cursor_plugin', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "profile = providers.REGISTRY['cursor']",
            "print(json.dumps({'name': profile.name, 'base_url': profile.base_url, 'auth_type': profile.auth_type}))",
        ]
    )
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(fake_core)}

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "name": "cursor",
        "base_url": "http://127.0.0.1:8765/v1",
        "auth_type": "api_key",
    }
