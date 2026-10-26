"""Double-click launcher: resume in the background or show an existing worker."""
import subprocess
import sys

from download_data import ROOT
from prepare_data import downloader_running


def main():
    if downloader_running():
        print("Dataset download is already running. No second download was started.", flush=True)
    else:
        local = ROOT / ".local"
        local.mkdir(exist_ok=True)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        # Append logs so a restart keeps the previous diagnostic information.
        with (local / "download-worker.log").open("ab") as output, (local / "download-worker-error.log").open("ab") as errors:
            worker = subprocess.Popen(
                [sys.executable, "-u", "scripts/download_data.py", "--set", "all", "--workers", "3"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                creationflags=flags, close_fds=True,
            )
        (local / "download-worker.pid").write_text(str(worker.pid), encoding="ascii")
        print(f"Background download started (PID {worker.pid}). Partial files will be resumed.", flush=True)
    print("You can close this window. Keep the computer awake and connected to the internet.", flush=True)
    print("Use check_status.cmd to check progress. Use prepare_data.cmd for archive extraction.", flush=True)
    subprocess.run([sys.executable, "scripts/status.py"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
