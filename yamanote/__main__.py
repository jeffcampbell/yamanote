"""python3 -m yamanote [--dashboard] [--dashboard-port PORT]"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys

from . import settings
from .factory import Factory
from .store import Store


def _acquire_pid_lock() -> None:
    path = settings.PID_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = int(path.read_text().strip())
            os.kill(old, 0)
        except (ValueError, OSError):
            path.unlink(missing_ok=True)  # stale or malformed
        else:
            sys.exit(f"Yamanote is already running (PID {old}).")
    path.write_text(str(os.getpid()))
    atexit.register(lambda: path.exists() and path.read_text().strip() == str(os.getpid()) and path.unlink())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yamanote", description="Yamanote — a software factory on the loop line")
    parser.add_argument("--dashboard", action="store_true", help="serve the dashboard on port 8080")
    parser.add_argument("--dashboard-port", type=int, default=0, metavar="PORT", help="serve the dashboard on PORT")
    parser.add_argument("--host", default=None, help=f"dashboard bind address (default {settings.DASHBOARD_HOST})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if not settings.openrouter_key():
        sys.exit("OPENROUTER_API_KEY is not set (environment, ./.env, or ~/development/.env).")
    _acquire_pid_lock()

    def _sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    factory = Factory(Store(settings.DB_PATH))
    port = args.dashboard_port or (8080 if args.dashboard else settings.DASHBOARD_PORT)
    if port:
        from .dashboard import start_dashboard
        start_dashboard(factory, port, args.host)
    factory.run_forever()
    # Don't let worker threads stuck in a slow LLM call hold the process open:
    # anything interrupted is re-queued by startup recovery.
    logging.shutdown()
    settings.PID_FILE.unlink(missing_ok=True)
    os._exit(0)


if __name__ == "__main__":
    main()
