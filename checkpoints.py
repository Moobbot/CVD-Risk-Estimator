"""The checkpoint files of the CVD service, and how the service gets them.

The service loads two files from its checkpoint folder (config.FOLDERS["CHECKPOINT"], /app/checkpoint
in the image). docker-compose mounts a HOST folder there (./CVD-Risk-Estimator/checkpoint): the
mount hides whatever the image has at that path, and on a fresh checkout the folder is empty. So:

- at image build, `python checkpoints.py --seed /app/checkpoint-seed` downloads the files into a
  folder nothing is mounted on (the "seed") and checks them;
- at service start, ensure_checkpoints() fills the mounted folder: a file that is already there is
  used as it is and NEVER replaced (a hospital's own copy); a missing file is copied from the seed;
  if the image has no seed, it is downloaded and checked.

Standard library only: this module runs at build time before the packages matter, and its tests run
without torch.
"""
import argparse
import glob
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
from typing import Callable, Dict, Iterable, NamedTuple, Tuple

_CHUNK = 1024 * 1024
_TIMEOUT_SECONDS = 60

# The image's own copy: next to this file (/app/checkpoint-seed), where no mount hides it.
SEED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint-seed")


class Checkpoint(NamedTuple):
    name: str
    sha256: str  # of the file the reference results were measured with
    urls: Tuple[str, ...]  # tried in order


# setup.py keeps the same addresses for running by hand outside Docker (tests/test_checkpoints.py
# checks that the two lists agree). `dl=1` makes Dropbox send the file instead of a web page.
CHECKPOINTS: Tuple[Checkpoint, ...] = (
    Checkpoint(
        name="retinanet_heart.pt",
        sha256="dccf38ef25b478dcb77a2d86a4ea4fd3a6beccd9f9776c648d9edd42da39982d",
        urls=(
            "https://www.dropbox.com/scl/fi/awfnv4elf1d9y9ca9kg8c/retinanet_heart.pt?rlkey=6exxr989ww6zs0cvosepw84sb&st=rpkioyoz&dl=1",
        ),
    ),
    Checkpoint(
        name="NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm",
        sha256="5d591131d18c9b7e23d3235f82f5550261d14b968d9f4950c2e76c9c5ec0fc1e",
        urls=(
            "https://www.dropbox.com/scl/fi/egwtns5xrbasg1i6dse19/NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm?rlkey=8csdx2h4dcoxwfo03k59qla94&st=kc1pa3t3&dl=1",
        ),
    ),
)

Fetch = Callable[[str, str], None]
_log = logging.getLogger("checkpoints")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    """Save `url` to `dest`. Raises on any network or HTTP error."""
    request = urllib.request.Request(url, headers={"User-Agent": "cvd-risk-estimator"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, _CHUNK)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _remove_stale_partials(dest: str) -> None:
    """Temporary files of an install that was killed half-way (ours only: `<name>.<random>.partial`)."""
    folder, name = os.path.split(dest)
    for stale in glob.glob(os.path.join(glob.escape(folder), glob.escape(name) + ".*.partial")):
        _remove(stale)


def _sync(path: str) -> None:
    """Flush a file (or a folder entry) to disk: a power cut must not leave an empty file under the real name."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _install(write: Callable[[str], None], dest: str, sha256: str) -> str:
    """Write to a temporary file of its own next to `dest`, and give it the real name only if it is
    the expected file AND the name is still free. Returns:

    installed  the file is now under the real name
    different  what was written is not the expected file: nothing installed
    exists     something took the real name meanwhile (e.g. an operator copied a file in): left alone

    Never replaces or removes anything under the real name. Raises what `write` raises.
    """
    folder, name = os.path.split(dest)
    fd, partial = tempfile.mkstemp(prefix=name + ".", suffix=".partial", dir=folder)
    os.close(fd)
    try:
        write(partial)
        if file_sha256(partial) != sha256:
            return "different"
        os.chmod(partial, 0o644)  # mkstemp makes it owner-only
        _sync(partial)
        try:
            os.link(partial, dest)  # atomic, and fails when the name is taken
        except FileExistsError:
            return "exists"
        except OSError:
            # A file system without hard links: rename after a last look (not atomic there).
            if os.path.lexists(dest):
                return "exists"
            os.rename(partial, dest)
        _sync(folder)
        return "installed"
    finally:
        _remove(partial)


def _download_one(spec: Checkpoint, dest: str, fetch: Fetch, log: logging.Logger) -> str:
    """'downloaded', 'exists' (the name was taken meanwhile), 'different' (something was downloaded,
    but not the checkpoint) or 'unreachable'."""
    outcome = "unreachable"
    for url in spec.urls:
        host = url.split("/")[2]
        try:
            log.info("Downloading %s from %s ...", spec.name, host)
            result = _install(lambda partial, url=url: fetch(url, partial), dest, spec.sha256)
            if result == "installed":
                return "downloaded"
            if result == "exists":
                return "exists"
            outcome = "different"
            log.error("What %s sent is not %s (a download cut short, or an error page).", host, spec.name)
        except Exception as e:  # network, HTTP, disk: try the next address
            log.warning("Could not download %s from %s: %s", spec.name, host, e)
    return outcome


def ensure_checkpoints(
    dest_dir: str,
    seed_dir: str = SEED_DIR,
    specs: Iterable[Checkpoint] = CHECKPOINTS,
    fetch: Fetch = download,
    log: logging.Logger = _log,
) -> Dict[str, str]:
    """Make sure every checkpoint file is in `dest_dir`. Returns, per file name:

    present     already there, and it is the validated file
    unexpected  already there, but another file: used as it is, and a warning is logged
    seeded      was missing, copied from the image's seed folder
    downloaded  was missing, downloaded and verified
    missing     could not be provided (logged); the caller's loading code reports the consequence
    """
    status: Dict[str, str] = {}
    os.makedirs(dest_dir, exist_ok=True)
    for spec in specs:
        dest = os.path.join(dest_dir, spec.name)
        try:
            status[spec.name] = _ensure_one(spec, dest, seed_dir, fetch, log)
        except Exception as e:  # e.g. the folder is read-only: one file must not stop the other
            log.error("Could not provide %s: %s", spec.name, e)
            status[spec.name] = "missing"
    return status


def _existing(spec: Checkpoint, dest: str, log: logging.Logger) -> str:
    """A file under the real name is used as it is, whatever it holds."""
    if file_sha256(dest) == spec.sha256:
        return "present"
    log.warning(
        "%s in the checkpoint folder is not the file this release was validated with. It is used as "
        "it is; to go back to the validated file, move it away and restart the service.",
        spec.name,
    )
    return "unexpected"


def _ensure_one(spec: Checkpoint, dest: str, seed_dir: str, fetch: Fetch, log: logging.Logger) -> str:
    if os.path.isfile(dest):
        return _existing(spec, dest, log)
    if os.path.lexists(dest):
        log.error("%s in the checkpoint folder is not a readable file (a broken link?): left as it is.", spec.name)
        return "missing"
    _remove_stale_partials(dest)

    seed_file = os.path.join(seed_dir, spec.name)
    if os.path.isfile(seed_file):
        result = _install(lambda partial: shutil.copyfile(seed_file, partial), dest, spec.sha256)
        if result == "installed":
            log.info("%s was missing from the checkpoint folder: copied from the image.", spec.name)
            return "seeded"
        if result == "exists":
            return _existing(spec, dest, log)
        log.error("The image's copy of %s is damaged: not used.", spec.name)

    outcome = _download_one(spec, dest, fetch, log)
    if outcome == "downloaded":
        log.info("%s was missing from the checkpoint folder: downloaded.", spec.name)
        return "downloaded"
    if outcome == "exists":
        return _existing(spec, dest, log)
    log.error(
        "%s is missing from the checkpoint folder and could not be provided. Copy the file into the "
        "folder mounted at /app/checkpoint and restart the service.",
        spec.name,
    )
    return "missing"


_WORDING = {
    "present": "present",
    "unexpected": "present but NOT the validated file (used as it is)",
    "seeded": "copied from the image",
    "downloaded": "downloaded",
    "missing": "MISSING",
}


def summary(status: Dict[str, str]) -> str:
    """One line for stdout: what ensure_checkpoints() found or did, and what to do about a missing file."""
    line = "Checkpoints: " + "; ".join(f"{name} {_WORDING.get(state, state)}" for name, state in status.items()) + "."
    if "missing" in status.values():
        line += " Copy the missing file(s) into the folder mounted at /app/checkpoint and restart the service."
    return line


def seed(
    seed_dir: str,
    specs: Iterable[Checkpoint] = CHECKPOINTS,
    fetch: Fetch = download,
    log: logging.Logger = _log,
) -> int:
    """Image build: download the files into the seed folder. Returns the exit code.

    A file that was downloaded but is not the validated checkpoint stops the build (1): the image
    must not carry another file. A file that could not be downloaded at all only warns (0): the
    build must not depend on reaching the download site; the service then downloads the file at
    start-up, unless the checkpoint folder already has it.
    """
    os.makedirs(seed_dir, exist_ok=True)
    code = 0
    for spec in specs:
        dest = os.path.join(seed_dir, spec.name)
        if os.path.isfile(dest) and file_sha256(dest) == spec.sha256:
            log.info("%s: already in the seed folder.", spec.name)
            continue
        _remove(dest)  # the image's own folder, during its build: a wrong leftover is not kept
        _remove_stale_partials(dest)
        outcome = _download_one(spec, dest, fetch, log)
        if outcome == "downloaded":
            log.info("%s: downloaded and verified.", spec.name)
        elif outcome in ("different", "exists"):
            log.error("ERROR: %s is not the expected checkpoint (download cut short or changed).", spec.name)
            code = 1
        else:
            log.warning(
                "WARNING: %s was not downloaded: the image has no copy of it. The service will download it "
                "at start-up unless the checkpoint folder already has it.",
                spec.name,
            )
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download the CVD checkpoints into the image's seed folder.")
    parser.add_argument("--seed", metavar="DIR", required=True, help="folder to download the checkpoints into")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return seed(args.seed)


if __name__ == "__main__":
    sys.exit(main())
