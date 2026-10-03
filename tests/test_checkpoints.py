"""checkpoints.py: how the service gets its two checkpoint files.

docker-compose mounts a host folder on /app/checkpoint, which hides the files the image has there
and starts empty on a fresh checkout. The service must then provide the files itself, without
replacing files that are already in the folder (a hospital's own copies).

Run: cd CVD-Risk-Estimator && python -m pytest tests/test_checkpoints.py -q
"""
import errno
import hashlib
import logging
import os
import re
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import checkpoints as ck  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD = b"the validated weights"
OTHER = b"some other weights"
BLOCK_PAGE = b"<html>Access to this site is blocked</html>"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spec(name="a.pt", data=GOOD, urls=("https://example.test/a.pt",)):
    return ck.Checkpoint(name=name, sha256=sha(data), urls=tuple(urls))


def write(path, data: bytes):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def read(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class NoDownload:
    """A fetch that must not be used. checkpoints.py catches whatever a fetch raises (a failed download
    is not fatal), so raising here would prove nothing: the calls are recorded and asserted empty."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append(url)
        raise OSError("no download expected in this test")


@pytest.fixture
def no_download():
    fetch = NoDownload()
    yield fetch
    assert fetch.calls == [], "the test downloaded although it must not"


def serving(content_by_url, calls=None):
    """A fetch function that writes what each URL 'returns'; an Exception value is raised."""

    def fetch(url, dest):
        if calls is not None:
            calls.append(url)
        answer = content_by_url[url]
        if isinstance(answer, Exception):
            raise answer
        with open(dest, "wb") as f:
            f.write(answer)

    return fetch


@pytest.fixture
def dirs(tmp_path):
    dest, seed = tmp_path / "checkpoint", tmp_path / "checkpoint-seed"
    dest.mkdir()
    seed.mkdir()
    return dest, seed


def leftovers(folder):
    """Files other than the checkpoint itself (a temporary file left behind would be one)."""
    return sorted(n for n in os.listdir(folder) if n != "a.pt")


# --- at service start -----------------------------------------------------------------------


def test_a_missing_file_is_copied_from_the_seed_without_downloading(dirs, no_download):
    dest, seed = dirs
    write(seed / "a.pt", GOOD)

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)

    assert status == {"a.pt": "seeded"}
    assert read(dest / "a.pt") == GOOD
    assert leftovers(dest) == []
    assert read(seed / "a.pt") == GOOD  # the seed stays: the next empty folder needs it too
    assert stat.S_IMODE(os.stat(dest / "a.pt").st_mode) == 0o644  # readable on the host, e.g. for a backup


def test_a_file_already_in_the_folder_is_never_replaced(dirs, no_download):
    """A hospital's own copy wins, even when it is not the file this release was validated with."""
    dest, seed = dirs
    write(dest / "a.pt", OTHER)
    write(seed / "a.pt", GOOD)

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)

    assert status == {"a.pt": "unexpected"}
    assert read(dest / "a.pt") == OTHER


def test_an_unexpected_file_is_reported_in_the_log(dirs, caplog, no_download):
    dest, seed = dirs
    write(dest / "a.pt", OTHER)
    with caplog.at_level(logging.WARNING):
        ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)
    assert any("a.pt" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_the_expected_file_already_in_the_folder_is_kept(dirs, no_download):
    dest, seed = dirs
    write(dest / "a.pt", GOOD)
    assert ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download) == {"a.pt": "present"}


def test_without_a_seed_the_file_is_downloaded_and_verified(dirs):
    dest, seed = dirs
    fetch = serving({"https://example.test/a.pt": GOOD})

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=fetch)

    assert status == {"a.pt": "downloaded"}
    assert read(dest / "a.pt") == GOOD
    assert leftovers(dest) == []


def test_a_download_that_is_not_the_checkpoint_is_not_installed(dirs):
    """A proxy answering with an error page must not become 'the checkpoint'."""
    dest, seed = dirs
    fetch = serving({"https://example.test/a.pt": BLOCK_PAGE})

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=fetch)

    assert status == {"a.pt": "missing"}
    assert os.listdir(dest) == []


def test_a_failed_download_leaves_nothing_behind(dirs):
    dest, seed = dirs
    fetch = serving({"https://example.test/a.pt": OSError("no route to host")})

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=fetch)

    assert status == {"a.pt": "missing"}
    assert os.listdir(dest) == []


def test_the_next_source_is_tried_when_one_fails(dirs):
    dest, seed = dirs
    calls = []
    urls = ("https://first.test/a.pt", "https://second.test/a.pt")
    fetch = serving({urls[0]: BLOCK_PAGE, urls[1]: GOOD}, calls)

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec(urls=urls)], fetch=fetch)

    assert status == {"a.pt": "downloaded"}
    assert calls == list(urls)
    assert read(dest / "a.pt") == GOOD


def test_a_damaged_seed_is_not_installed(dirs):
    dest, seed = dirs
    write(seed / "a.pt", GOOD[:5])  # cut short
    fetch = serving({"https://example.test/a.pt": GOOD})

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=fetch)

    assert status == {"a.pt": "downloaded"}
    assert read(dest / "a.pt") == GOOD


def test_a_file_left_half_written_by_an_interrupted_start_is_replaced(dirs, no_download):
    dest, seed = dirs
    write(dest / "a.pt.k3x9.partial", GOOD[:3])
    write(seed / "a.pt", GOOD)

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)

    assert status == {"a.pt": "seeded"}
    assert read(dest / "a.pt") == GOOD
    assert leftovers(dest) == []


def test_a_file_that_appears_while_downloading_is_not_replaced(dirs):
    """E.g. the operator copies the hospital's file into the folder while the service is still
    downloading: that file must win, like any file that was there before the start."""
    dest, seed = dirs

    def fetch(url, partial):
        write(dest / "a.pt", OTHER)  # appears under the real name during the download
        with open(partial, "wb") as f:
            f.write(GOOD)

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=fetch)

    assert read(dest / "a.pt") == OTHER
    assert status == {"a.pt": "unexpected"}
    assert leftovers(dest) == []


def test_a_file_that_appears_while_copying_from_the_seed_is_not_replaced(dirs, no_download, monkeypatch):
    dest, seed = dirs
    write(seed / "a.pt", GOOD)
    real_copy = ck.shutil.copyfile

    def copy_then_someone_else_writes(src, dst, **kw):
        real_copy(src, dst, **kw)
        write(dest / "a.pt", OTHER)

    monkeypatch.setattr(ck.shutil, "copyfile", copy_then_someone_else_writes)

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)

    assert read(dest / "a.pt") == OTHER
    assert status == {"a.pt": "unexpected"}
    assert leftovers(dest) == []


def test_two_starts_at_once_do_not_share_a_temporary_file(dirs, no_download):
    """Each install writes to its own temporary name, so one cannot take the other's away."""
    dest, seed = dirs
    write(seed / "a.pt", GOOD)
    seen = []
    real_install = ck._install

    def spying(write_to, target, sha256):
        def recording(partial):
            seen.append(os.path.basename(partial))
            write_to(partial)

        return real_install(recording, target, sha256)

    ck._install, saved = spying, ck._install
    try:
        ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)
        os.remove(dest / "a.pt")
        ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)
    finally:
        ck._install = saved
    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(n.startswith("a.pt.") and n.endswith(".partial") for n in seen)


def test_a_file_system_without_hard_links_still_gets_the_file(dirs, no_download, monkeypatch):
    dest, seed = dirs
    write(seed / "a.pt", GOOD)

    def no_links(src, dst):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(ck.os, "link", no_links)

    assert ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download) == {"a.pt": "seeded"}
    assert read(dest / "a.pt") == GOOD
    assert leftovers(dest) == []


def test_something_that_is_not_a_file_under_the_name_is_left_alone(dirs, no_download):
    dest, seed = dirs
    write(seed / "a.pt", GOOD)
    os.symlink(str(dest / "nowhere"), str(dest / "a.pt"))  # a dangling link someone made

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download)

    assert status == {"a.pt": "missing"}
    assert os.path.islink(dest / "a.pt")


def test_the_folder_is_created_when_it_does_not_exist(tmp_path, no_download):
    seed = tmp_path / "seed"
    write(seed / "a.pt", GOOD)
    dest = tmp_path / "not" / "yet"

    assert ck.ensure_checkpoints(str(dest), str(seed), specs=[spec()], fetch=no_download) == {"a.pt": "seeded"}
    assert read(dest / "a.pt") == GOOD


def test_no_seed_folder_at_all_is_not_an_error(tmp_path):
    """Outside Docker there is no seed folder: the file is downloaded."""
    dest = tmp_path / "checkpoint"
    fetch = serving({"https://example.test/a.pt": GOOD})

    status = ck.ensure_checkpoints(str(dest), str(tmp_path / "absent"), specs=[spec()], fetch=fetch)

    assert status == {"a.pt": "downloaded"}


def test_each_file_is_handled_on_its_own(dirs, no_download):
    dest, seed = dirs
    write(dest / "a.pt", GOOD)
    write(seed / "b.pt", OTHER)
    specs = [spec(), spec(name="b.pt", data=OTHER, urls=("https://example.test/b.pt",))]

    status = ck.ensure_checkpoints(str(dest), str(seed), specs=specs, fetch=no_download)

    assert status == {"a.pt": "present", "b.pt": "seeded"}


def test_the_start_up_line_says_where_each_file_came_from():
    """Printed on stdout: outside dev the service's loggers write to files only, and
    `docker compose logs cvd` is where an operator looks."""
    line = ck.summary({"a.pt": "seeded", "b.ptm": "present"})
    assert line == "Checkpoints: a.pt copied from the image; b.ptm present."


def test_the_start_up_line_tells_what_to_do_about_a_missing_file():
    line = ck.summary({"a.pt": "missing", "b.ptm": "downloaded"})
    assert "a.pt MISSING" in line and "b.ptm downloaded" in line
    assert "/app/checkpoint" in line and "restart" in line


def test_the_start_up_line_flags_a_file_that_is_not_the_validated_one():
    line = ck.summary({"a.pt": "unexpected"})
    assert "a.pt present but NOT the validated file" in line
    assert "restart" not in line  # nothing to fix: the file is used as it is


# --- at image build (`python checkpoints.py --seed <dir>`) ----------------------------------


def test_build_puts_verified_files_in_the_seed_folder(tmp_path):
    seed = tmp_path / "seed"
    fetch = serving({"https://example.test/a.pt": GOOD})

    assert ck.seed(str(seed), specs=[spec()], fetch=fetch) == 0
    assert read(seed / "a.pt") == GOOD
    assert leftovers(seed) == []


def test_build_stops_when_a_downloaded_file_is_not_the_checkpoint(tmp_path):
    """As before this change: an image must not carry a file other than the validated one."""
    seed = tmp_path / "seed"
    fetch = serving({"https://example.test/a.pt": BLOCK_PAGE})

    assert ck.seed(str(seed), specs=[spec()], fetch=fetch) == 1
    assert not os.path.exists(seed / "a.pt")


def test_build_only_warns_when_a_file_cannot_be_downloaded(tmp_path, caplog):
    """As before this change: the build must not depend on reaching the download site."""
    seed = tmp_path / "seed"
    fetch = serving({"https://example.test/a.pt": OSError("blocked")})

    with caplog.at_level(logging.WARNING):
        assert ck.seed(str(seed), specs=[spec()], fetch=fetch) == 0
    assert not os.path.exists(seed / "a.pt")
    assert any("WARNING" in r.getMessage() and "a.pt" in r.getMessage() and "not downloaded" in r.getMessage() for r in caplog.records)


def test_build_does_not_keep_a_wrong_file_already_in_the_seed_folder(tmp_path):
    seed = tmp_path / "seed"
    write(seed / "a.pt", OTHER)
    fetch = serving({"https://example.test/a.pt": OSError("blocked")})

    assert ck.seed(str(seed), specs=[spec()], fetch=fetch) == 0
    assert not os.path.exists(seed / "a.pt")


def test_build_downloads_nothing_when_the_seed_folder_already_has_the_file(tmp_path, no_download):
    seed = tmp_path / "seed"
    write(seed / "a.pt", GOOD)

    assert ck.seed(str(seed), specs=[spec()], fetch=no_download) == 0
    assert read(seed / "a.pt") == GOOD


# --- the real download function ---------------------------------------------------------------


@pytest.fixture
def http_server(tmp_path):
    """A local web server: /a.pt answers GOOD, anything else 404."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/a.pt":
                self.send_response(200)
                self.send_header("Content-Length", str(len(GOOD)))
                self.end_headers()
                self.wfile.write(GOOD)
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_download_saves_what_the_address_sends(http_server, tmp_path):
    ck.download(f"{http_server}/a.pt", str(tmp_path / "out"))
    assert read(tmp_path / "out") == GOOD


def test_download_raises_on_an_http_error(http_server, tmp_path):
    with pytest.raises(Exception):
        ck.download(f"{http_server}/gone.pt", str(tmp_path / "out"))


def test_an_address_that_answers_404_counts_as_not_downloaded(http_server, dirs):
    dest, seed = dirs
    one = spec(urls=(f"{http_server}/gone.pt", f"{http_server}/a.pt"))

    assert ck.ensure_checkpoints(str(dest), str(seed), specs=[one]) == {"a.pt": "downloaded"}
    assert read(dest / "a.pt") == GOOD


# --- the real list, and the files that must agree with it ------------------------------------


def test_the_real_list_names_the_two_files_the_service_loads():
    names = {c.name for c in ck.CHECKPOINTS}
    assert names == {"retinanet_heart.pt", "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm"}
    for c in ck.CHECKPOINTS:
        assert re.fullmatch(r"[0-9a-f]{64}", c.sha256), c.name
        assert c.urls and all(u.startswith("https://") for u in c.urls), c.name


def _text(*parts):
    path = os.path.join(ROOT, *parts)
    if not os.path.exists(path):
        # Dockerfile and .dockerignore are not copied into the image: these checks run in the repository.
        pytest.skip(f"{'/'.join(parts)} is not in the image; run this test in the repository")
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_setup_py_downloads_from_the_same_addresses():
    """setup.py (run by hand outside Docker) keeps its own list: the two must not drift."""
    text = _text("setup.py")
    for c in ck.CHECKPOINTS:
        assert f'"{c.name}": "{c.urls[0]}"' in text, c.name


def test_the_image_keeps_its_copy_where_no_mount_hides_it():
    dockerfile = _text("Dockerfile")
    assert "RUN python checkpoints.py --seed /app/checkpoint-seed" in dockerfile
    assert os.path.basename(ck.SEED_DIR) == "checkpoint-seed"
    assert os.path.dirname(ck.SEED_DIR) == ROOT  # /app in the image


def test_weights_in_the_host_folder_are_not_copied_into_the_image():
    """Once the service has run, ./checkpoint on the host holds the weights; `COPY . .` must skip them."""
    lines = [line.strip() for line in _text(".dockerignore").splitlines()]
    assert "checkpoint/*" in lines
    assert "!checkpoint/.gitignore" in lines and "!checkpoint/README.md" in lines
