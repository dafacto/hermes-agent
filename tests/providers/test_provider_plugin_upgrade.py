"""A provider plugin upgraded in place reaches a process that already imported it.

Replays the Lakpa incident (25-26 Oct 2026): the Oi connector replaced its
``plugins/model-providers/oi-claude-code`` folder (rename-swap) with a version that also registers
``oi-claude-directsdk``, and the default model moved to that provider. The long-running gateway had
imported the old folder, skipped the new one as "already imported", and every cron job following the
default was blocked with ``Unknown provider 'oi-claude-directsdk'`` until a restart.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

_PLUGIN = '''
from providers import register_provider
from providers.base import ProviderProfile

for _name in {names!r}:
    register_provider(ProviderProfile(name=_name, display_name=_name, api_mode="chat_completions",
                                      auth_type="external_process", env_vars=(),
                                      base_url="process://" + _name))
'''

_PROBE = '''
import json, os, sys
from pathlib import Path
import providers
providers._PLUGIN_DIR_STAMP_TTL_SECONDS = 0.0
from hermes_cli.auth import resolve_provider

def known(name):
    try:
        return resolve_provider(name) == name
    except Exception:
        return False

home = Path(os.environ["HERMES_HOME"])
parent = home / "plugins" / "model-providers"
before = {"one": known("acme-one"), "two": known("acme-two")}
# The connector's install(): stage beside the target, move the old folder out, rename the new in.
staged = parent / ".acme-stage"
staged.mkdir()
(staged / "plugin.yaml").write_text("name: acme\\nkind: model-provider\\nversion: 2.0.0\\n")
(staged / "__init__.py").write_text(sys.argv[1])
os.rename(parent / "acme", home / "acme-old")
os.rename(staged, parent / "acme")
after = {"one": known("acme-one"), "two": known("acme-two")}
# Unchanged folder: no further re-import (the module object stays the same).
module = sys.modules[[n for n in sys.modules if n.endswith("_acme")][0]]
known("acme-two")
stable = sys.modules[[n for n in sys.modules if n.endswith("_acme")][0]] is module
print(json.dumps({"before": before, "after": after, "stable": stable}))
'''


def test_in_place_upgrade_registers_the_new_provider_without_a_restart(tmp_path):
    home = tmp_path / "home"
    plugin = home / "plugins" / "model-providers" / "acme"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: acme\nkind: model-provider\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text(_PLUGIN.format(names=("acme-one",)))
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: acme-one\n")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HERMES_", "_HERMES_")) and not k.endswith(("_API_KEY", "_TOKEN"))}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               PYTHONDONTWRITEBYTECODE="1")

    result = subprocess.run(
        [sys.executable, "-c", _PROBE, _PLUGIN.format(names=("acme-one", "acme-two"))],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["before"] == {"one": True, "two": False}
    assert report["after"] == {"one": True, "two": True}, (
        "a provider added by an in-place plugin upgrade stayed unknown to the running process")
    assert report["stable"]
