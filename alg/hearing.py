# wait_and_run.py
import subprocess
import time
import psutil  # pip install psutil


def is_launcher_running(name="launcher.py"):
    for proc in psutil.process_iter(["cmdline"]):
        cmd = " ".join(proc.info["cmdline"])
        if name in cmd:
            return True
    return False


print("waiting")
while is_launcher_running():
    time.sleep(20)

print("starting")
subprocess.call("python launcher.py", shell=True)

print("finishing")
