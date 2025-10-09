import os
import json
import debugpy
import socket
import random

def update_vscode_launch_file(host: str, port: int):
    """Update the .vscode/launch.json file with the new host and port."""
    launch_file_path = ".vscode/launch.json"
    # Desired configuration
    new_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "bash_debug",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": host,
                    "port": port
                },
                "justMyCode": False
            },
        ]
    }

    # Ensure the .vscode directory exists
    if not os.path.exists(".vscode"):
        os.makedirs(".vscode")

    # Write the updated configuration to launch.json
    with open(launch_file_path, "w") as f:
        json.dump(new_config, f, indent=4)
    print(f"Updated {launch_file_path} with host: {host} and port: {port}")

def is_port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def setup_debug(is_main_process=True, max_retries=10, port_range=(10000, 20000)):
    if is_main_process:
        host = os.environ['SLURM_NODELIST'].split(',')[0]

        for _ in range(max_retries):
            port = random.randint(*port_range)
            try:
                if is_port_in_use(host, port):
                    print(f"Port {port} is already in use, trying another...")
                    continue

                # 更新 launch.json
                update_vscode_launch_file(host, port)

                print("master_addr = ", host)
                debugpy.listen((host, port))
                print(f"Waiting for debugger attach at port {port}...", flush=True)
                debugpy.wait_for_client()
                print("Debugger attached", flush=True)
                return
            except Exception as e:
                print(f"Failed to bind to port {port}: {e}")

        raise RuntimeError("Could not find a free port for debugpy after several attempts.")