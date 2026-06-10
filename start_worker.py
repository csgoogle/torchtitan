#!/usr/bin/env python3
# start_worker.py - Monarch worker startup entrypoint
import os
import time

def main():
    print("@@@ Spawning Monarch Worker Agent...", flush=True)
    
    # 1. Initialize logs directory
    os.makedirs("/tmp/root", exist_ok=True)
    log_file = "/tmp/root/monarch_log.log"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("")

    # 2. Configure Monarch transport
    from monarch.config import configure, ChannelTransport
    configure(default_transport=ChannelTransport.TcpWithHostname)
    
    # 3. Retrieve pod IP and start listening
    pod_ip = os.getenv("POD_IP", "127.0.0.1")
    bind_address = f"tcp://{pod_ip}:26600"
    print(f"@@@ Listening on: {bind_address}", flush=True)

    from monarch.actor import run_worker_loop_forever
    try:
        run_worker_loop_forever(ca="trust_all_connections", address=bind_address)
    except Exception as e:
        print(f"@@@ Worker Loop terminated with exception: {e}", flush=True)
        # Sleep to allow debugging of crashed worker pods instead of immediate crash loops
        time.sleep(86400)

if __name__ == "__main__":
    main()
