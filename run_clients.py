import subprocess
import time

def run_clients(num_clients=10):
    processes = []
    try:
        for client_id in range(num_clients):
            command = [
                "python",
                "src/client_socket.py",
                "--client_id",
                str(client_id),
                "--num_clients",
                str(num_clients),
            ]
            print(f"[Main Script] Starting client {client_id}...")
            # Start each client as a separate process
            process = subprocess.Popen(command, stderr=subprocess.PIPE)
            processes.append(process)

        print("[Main Script] Waiting for all clients to complete...")
        # Wait for all processes to complete
        # Monitor processes until all are completed
        while processes:
            for process in processes:
                retcode = process.poll()
                if retcode is not None:  # Process has finished
                    if retcode == 0:
                        print(f"[Main Script] Client process {process.pid} finished successfully.")
                    else:
                        error_message = process.stderr.read().decode('utf-8') 
                        print(f"[Main Script] Client process {process.pid} failed with exit code {retcode}.")
                        print(f"[Main Script] Error message: {error_message}")
                    print(f"[Main Script] Client process {process.pid} finished with exit code {retcode}.")
                    processes.remove(process)
            time.sleep(1)  # Avoid busy-waiting

        print("[Main Script] All clients have completed successfully.")
    except KeyboardInterrupt:
        print("\n[Main Script] Keyboard interrupt received. Terminating clients...")
        # Terminate all running processes on interrupt
        for process in processes:
            process.terminate()
    except Exception as e:
        print(f"[Main Script] An unexpected error occurred: {e}")
    finally:
        print("[Main Script] Ensuring all clients are stopped...")
        for process in processes:
            if process.poll() is None:  # If process is still running
                process.terminate()
                process.wait()  # Ensure termination is complete
        print("[Main Script] All clients have been stopped. Exiting program.")

if __name__ == "__main__":
    run_clients(num_clients=10)
