"""
VERITAS-X Launcher Helper.
Provides robust port inspection, readiness polling, browser launching,
and process PID tracking for START_VERITAS_X.bat and STOP_VERITAS_X.bat.
"""
import os
import sys
import time
import json
import socket
import urllib.request
import urllib.error
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PID_FILE = os.path.join(REPO_ROOT, "tmp", "runtime_pids.json")


API_PORT = 8011
FE_PORT = 5173


def is_port_in_use(port: int = API_PORT, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def check_api_health(url: str = f"http://127.0.0.1:{API_PORT}/api/health", timeout: float = 2.0):
    """Returns (http_ready: bool, model_ready: bool, data: dict)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VERITAS-X-Launcher"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                http_ok = data.get("status") == "ok"
                model_ok = bool(data.get("model_ready", False))
                return http_ok, model_ok, data
    except Exception:
        pass
    return False, False, {}


def check_frontend_ready(url: str = f"http://127.0.0.1:{FE_PORT}", timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VERITAS-X-Launcher"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 304)
    except Exception:
        return False


def get_pid_listening_on_port(port: int) -> list:
    """Finds PIDs listening on a given port using netstat on Windows."""
    pids = []
    try:
        output = subprocess.check_output(f"netstat -ano -p tcp", shell=True).decode("utf-8", errors="ignore")
        for line in output.splitlines():
            line = line.strip()
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) > 0:
                        pids.append(int(pid))
    except Exception:
        pass
    return sorted(list(set(pids)))


def cmd_preflight():
    """Checks port availability and existing server status."""
    api_occupied = is_port_in_use(API_PORT)
    api_http, api_model, _ = check_api_health() if api_occupied else (False, False, {})

    fe_occupied = is_port_in_use(FE_PORT)
    fe_ready = check_frontend_ready() if fe_occupied else False

    status = {
        "api_port": API_PORT,
        "api_occupied": api_occupied,
        "api_healthy": api_http,
        "model_ready": api_model,
        "fe_port": FE_PORT,
        "fe_occupied": fe_occupied,
        "fe_ready": fe_ready,
    }
    print(json.dumps(status))


def cmd_wait_ready(timeout_sec: int = 40):
    """Waits for both backend HTTP, backend model readiness, and frontend."""
    print("[*] Waiting for VERITAS-X services and AI verification engine...")
    start_time = time.time()
    api_http = False
    api_model = False
    fe_ready = False

    while time.time() - start_time < timeout_sec:
        if not api_http or not api_model:
            api_http, api_model, _ = check_api_health()
        if not fe_ready:
            fe_ready = check_frontend_ready()

        if api_http and api_model and fe_ready:
            elapsed = time.time() - start_time
            print(f"\n[+] All services and AI verification engine confirmed ready in {elapsed:.1f}s!")

            # Record listening PIDs
            os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
            api_pids = get_pid_listening_on_port(API_PORT)
            fe_pids = get_pid_listening_on_port(FE_PORT)
            pid_data = {
                "started_at": time.time(),
                "api_pids": api_pids,
                "fe_pids": fe_pids,
            }
            try:
                with open(PID_FILE, "w", encoding="utf-8") as f:
                    json.dump(pid_data, f, indent=2)
            except Exception:
                pass
            return 0

        dots = "." * (int(time.time() - start_time) % 4)
        backend_status = "READY" if (api_http and api_model) else ("WARMING MODEL" if api_http else "PENDING")
        frontend_status = "READY" if fe_ready else "PENDING"
        sys.stdout.write(f"\r    Waiting for Backend ({API_PORT}) [{backend_status}] | Frontend ({FE_PORT}) [{frontend_status}] {dots:<4}")
        sys.stdout.flush()
        time.sleep(1.0)

    print("\n[!] Timeout waiting for services to become ready.")
    if not api_http:
        print(f"    - Backend on http://127.0.0.1:{API_PORT} did not respond to /api/health.")
    elif not api_model:
        print("    - Backend HTTP responded, but model engine is still initializing.")
    if not fe_ready:
        print(f"    - Frontend on http://127.0.0.1:{FE_PORT} did not respond.")
    return 1


def cmd_stop():
    """Safely terminates only VERITAS-X API and Frontend processes."""
    print("[*] Stopping VERITAS-X processes...")
    stopped_count = 0

    pids_to_kill = set()
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            pids_to_kill.update(data.get("api_pids", []))
            pids_to_kill.update(data.get("fe_pids", []))
        except Exception:
            pass

    # Also check current listening ports (8011 and 5173 only)
    current_api = get_pid_listening_on_port(API_PORT)
    current_fe = get_pid_listening_on_port(FE_PORT)
    pids_to_kill.update(current_api)
    pids_to_kill.update(current_fe)

    for pid in pids_to_kill:
        if pid <= 4:
            continue
        try:
            p_out = subprocess.check_output(f'tasklist /FI "PID eq {pid}" /FO CSV /NH', shell=True).decode("utf-8", errors="ignore")
            p_name = p_out.split(",")[0].replace('"', '').strip().lower()
            if any(match in p_name for match in ["python", "node", "cmd"]):
                subprocess.call(f"taskkill /PID {pid} /F /T >nul 2>&1", shell=True)
                print(f"    Terminated PID {pid} ({p_name})")
                stopped_count += 1
        except Exception as e:
            pass

    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except Exception:
            pass

    if stopped_count > 0:
        print(f"[+] Successfully terminated {stopped_count} VERITAS-X process(es).")
    else:
        print(f"[*] No active VERITAS-X server processes were found on ports {API_PORT} or {FE_PORT}.")
    return 0


def cmd_preflight_tokens():
    """Prints space-separated preflight status tokens for safe batch consumption."""
    api_occupied = is_port_in_use(API_PORT)
    api_http, api_model, _ = check_api_health() if api_occupied else (False, False, {})
    fe_occupied = is_port_in_use(FE_PORT)
    fe_ready = check_frontend_ready() if fe_occupied else False
    print(f"{api_occupied} {api_http} {fe_occupied} {fe_ready}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python launcher_helper.py [preflight|preflight-tokens|wait-ready|stop]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "preflight":
        cmd_preflight()
    elif cmd in ("preflight-tokens", "preflight_tokens"):
        cmd_preflight_tokens()
    elif cmd == "wait-ready":
        sys.exit(cmd_wait_ready())
    elif cmd == "stop":
        sys.exit(cmd_stop())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

