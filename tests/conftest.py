"""Ensure that installed extension is up-to-date."""
from pathlib import Path
import importlib.util
import subprocess


ROOT = Path(__file__).parent.parent


def pytest_sessionstart():
    """Detect an out of date extension and reinstall prior to testing."""
    spec = importlib.util.find_spec('wasabigeom')
    installed_ext = Path(spec.origin)
    if not installed_ext.exists():
        print("Extension is not installed, installing...")
        return install_ext()

    sources = ROOT.glob('*.pyx')
    ext_mtime = installed_ext.stat().st_mtime
    need_install = any(
        source.stat().st_mtime > ext_mtime
        for source in sources
    )
    if need_install:
        print("Extension is out of date, installing...")
        install_ext()


def install_ext():
    """Re-install the extension."""
    subprocess.check_call(['pip', 'install', ROOT])
