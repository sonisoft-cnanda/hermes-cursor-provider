"""Install the standalone provider shim into a Hermes home."""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .profile import DEFAULT_BRIDGE_BASE_URL, FALLBACK_MODELS


def _render_plugin_init() -> str:
    return f'''"""Standalone Cursor provider profile for Hermes Agent."""\n\nfrom providers import register_provider\nfrom providers.base import ProviderProfile\n\ncursor = ProviderProfile(\n    name="cursor",\n    aliases=("cursor-agent", "cursor-cli", "cursor-sub", "cursor-subscription"),\n    display_name="Cursor",\n    description="Hermes-controlled inference through a local Cursor CLI bridge",\n    signup_url="https://cursor.com/dashboard/integrations",\n    api_mode="chat_completions",\n    env_vars=("CURSOR_BRIDGE_API_KEY",),\n    base_url={DEFAULT_BRIDGE_BASE_URL!r},\n    auth_type="api_key",\n    fallback_models={FALLBACK_MODELS!r},\n    supports_health_check=True,\n)\n\nregister_provider(cursor)\n'''


PLUGIN_INIT = _render_plugin_init()
PLUGIN_MANIFEST = """name: cursor
kind: model-provider
version: 0.2.0
description: Hermes-controlled inference through an authenticated Cursor CLI bridge
"""
_ENV_KEY = "CURSOR_BRIDGE_API_KEY"
_OWNED_FILES = ("__init__.py", "plugin.yaml")


@dataclass(frozen=True)
class InstallResult:
    plugin_dir: Path
    env_path: Path
    token: str
    backup_dir: Path | None = None


def _validate_token(token: str) -> str:
    cleaned = token.strip()
    if not cleaned or any(character in cleaned for character in "\r\n\x00"):
        raise ValueError("Bridge token must be a non-empty single-line value")
    if len(cleaned) < 43:
        raise ValueError("Bridge token must contain at least 43 characters")
    if len(set(cleaned)) < 8:
        raise ValueError("Bridge token is too weak; generate a new random token")
    return cleaned


def _reject_managed_symlinks(home: Path, *, include_env: bool) -> None:
    """Reject links in package-managed paths before any read, write, or removal."""
    plugin_root = home / "plugins"
    provider_root = plugin_root / "model-providers"
    plugin_dir = provider_root / "cursor"
    paths = [
        plugin_root,
        provider_root,
        plugin_dir,
        *(plugin_dir / filename for filename in _OWNED_FILES),
        plugin_dir / "__pycache__",
    ]
    if include_env:
        paths.append(home / ".env")
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Refusing to follow symlink in managed path: {path}")


def _atomic_write(path: Path, content: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        if mode is not None:
            os.chmod(path, mode)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _existing_token(content: str) -> str | None:
    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if not line.startswith(f"{_ENV_KEY}="):
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return _validate_token(value)
    return None


def _upsert_token(env_path: Path, requested_token: str) -> str:
    existing_content = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    current = _existing_token(existing_content)
    if current:
        os.chmod(env_path, 0o600)
        return current
    content = existing_content
    if content and not content.endswith("\n"):
        content += "\n"
    content += f"{_ENV_KEY}={requested_token}\n"
    _atomic_write(env_path, content, mode=0o600)
    return requested_token


def install_plugin(
    hermes_home: str | Path,
    *,
    token: str | None = None,
    force: bool = False,
) -> InstallResult:
    """Install provider-owned files and preserve any existing local token."""
    home = Path(hermes_home).expanduser().resolve()
    requested_token = _validate_token(token or secrets.token_urlsafe(32))
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    env_path = home / ".env"
    _reject_managed_symlinks(home, include_env=True)
    plugin_existed = plugin_dir.exists()
    env_existed = env_path.exists()
    env_content = env_path.read_text(encoding="utf-8") if env_existed else ""
    env_mode = (env_path.stat().st_mode & 0o777) if env_existed else None
    expected = {
        "__init__.py": PLUGIN_INIT,
        "plugin.yaml": PLUGIN_MANIFEST,
    }
    conflicts = [
        filename
        for filename, content in expected.items()
        if (plugin_dir / filename).exists()
        and (plugin_dir / filename).read_text(encoding="utf-8") != content
    ]
    if conflicts and not force:
        joined = ", ".join(conflicts)
        raise FileExistsError(
            f"Refusing to overwrite existing Cursor plugin files ({joined}); "
            "inspect them first or rerun install with force=True"
        )
    existing_token = _existing_token(env_content) if env_existed else None
    effective_token = existing_token or requested_token
    backup_dir: Path | None = None
    if conflicts and force:
        plugin_dir.parent.mkdir(parents=True, exist_ok=True)
        backup_dir = Path(
            tempfile.mkdtemp(prefix="cursor.backup-", dir=str(plugin_dir.parent))
        )
        backup_dir.rmdir()
        os.replace(plugin_dir, backup_dir)
    try:
        plugin_dir.mkdir(parents=True, exist_ok=True)
        _reject_managed_symlinks(home, include_env=True)
        _atomic_write(plugin_dir / "__init__.py", PLUGIN_INIT, mode=0o644)
        _atomic_write(plugin_dir / "plugin.yaml", PLUGIN_MANIFEST, mode=0o644)
        effective_token = _upsert_token(env_path, requested_token)
    except Exception:
        if backup_dir is not None:
            shutil.rmtree(plugin_dir, ignore_errors=True)
            os.replace(backup_dir, plugin_dir)
        elif not plugin_existed:
            shutil.rmtree(plugin_dir, ignore_errors=True)
        if env_existed:
            try:
                _atomic_write(env_path, env_content, mode=env_mode)
            except OSError as rollback_error:
                raise RuntimeError(
                    "Plugin installation failed and the Hermes .env rollback also failed"
                ) from rollback_error
        else:
            try:
                env_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return InstallResult(
        plugin_dir=plugin_dir,
        env_path=env_path,
        token=effective_token,
        backup_dir=backup_dir,
    )


def uninstall_plugin(hermes_home: str | Path, *, force: bool = False) -> None:
    """Remove only files owned by this package; retain credentials and user files."""
    home = Path(hermes_home).expanduser().resolve()
    plugin_dir = home / "plugins" / "model-providers" / "cursor"
    _reject_managed_symlinks(home, include_env=False)
    expected = {
        "__init__.py": PLUGIN_INIT,
        "plugin.yaml": PLUGIN_MANIFEST,
    }
    modified = [
        filename
        for filename, content in expected.items()
        if (plugin_dir / filename).exists()
        and (plugin_dir / filename).read_text(encoding="utf-8") != content
    ]
    if modified and not force:
        joined = ", ".join(modified)
        raise FileExistsError(
            f"Refusing to remove modified Cursor plugin files ({joined}); "
            "inspect them first or rerun uninstall with force=True"
        )
    for filename in _OWNED_FILES:
        try:
            (plugin_dir / filename).unlink()
        except FileNotFoundError:
            pass
    pycache = plugin_dir / "__pycache__"
    if pycache.is_dir():
        shutil.rmtree(pycache)
    try:
        plugin_dir.rmdir()
    except (FileNotFoundError, OSError):
        pass
