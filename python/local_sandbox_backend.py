"""Shared coding-agent backend construction for the m2.3 sandbox scripts.

LangSmith's hosted sandbox requires a premium subscription.
Set LCA_BACKEND=langsmith to use it instead of the local backend
(default, no subscription needed, runs directly on this machine).
"""

import os
import shutil
import tempfile
from uuid import uuid4

USE_LANGSMITH_SANDBOX = os.environ.get("LCA_BACKEND", "local") == "langsmith"


def sandbox_path(name: str) -> str:
    """Build a file path to hand to the agent (in prompts) and to
    `backend.upload_files`/`download_files`, that works with whichever
    backend `create_backend` picked.

    LangSmith's sandbox is its own isolated filesystem, so paths must be
    absolute (e.g. "/chart.png") - that's the sandbox's root, not the host's.

    `LocalShellBackend` has no isolated root: code the agent runs via
    `execute()` is unrestricted and an absolute path would hit the real
    filesystem root (see `create_backend`). Relative paths (e.g. "chart.png")
    resolve under the backend's working directory for both the filesystem
    tools *and* executed shell commands, so that's what this returns locally.
    """
    return f"/{name}" if USE_LANGSMITH_SANDBOX else name


def create_backend(name_prefix: str):
    """Create a coding-agent backend and its cleanup callback.

    Args:
        name_prefix: Used to name the sandbox (LangSmith) or
        the temp directory (local) so multiple scripts don't collide.

    Returns:
        `(backend, cleanup)`.
        `backend` implements deepagents'
        `SandboxBackendProtocol` (`execute`/`read`/`write`/`upload_files`/
        `download_files`/...), the same interface for either choice.
        Always call `cleanup()` when done: it deletes the LangSmith sandbox
        or the local temp dir, so download anything you need to keep first.
    """
    if USE_LANGSMITH_SANDBOX:
        from deepagents.backends.langsmith import LangSmithSandbox
        from langsmith.sandbox import SandboxClient

        client = SandboxClient()
        ls_sandbox = client.create_sandbox(name=f"{name_prefix}-{uuid4().hex[:8]}")
        print(f"Sandbox: {ls_sandbox.name}  (id: {ls_sandbox.id})")
        backend = LangSmithSandbox(sandbox=ls_sandbox)
        return backend, lambda: client.delete_sandbox(ls_sandbox.name)

    from deepagents.backends import LocalShellBackend

    # Runs commands directly on this machine (no sandboxing/isolation),
    # rooted at this temp dir. virtual_mode=True scopes the filesystem tools
    # (write/read/edit/ls/upload_files/download_files) to that dir, even for
    # absolute paths the agent might use on its own. It has no effect on the
    # *relative* paths `sandbox_path` hands out above - those already resolve
    # under this dir either way - and, per LocalShellBackend's docstring, no
    # effect on `execute()`, which is never sandboxed: code the agent runs
    # itself can still reach any host path if it hardcodes an absolute one.
    tmp_dir = tempfile.mkdtemp(prefix=f"{name_prefix}-")
    print(f"Local backend: {tmp_dir}")
    backend = LocalShellBackend(root_dir=tmp_dir, virtual_mode=True, inherit_env=True)
    return backend, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)
