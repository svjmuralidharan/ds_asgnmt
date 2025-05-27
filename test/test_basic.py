import subprocess

def main():
    # Change 'your_script.py' to the name of your actual script!
    script = "your_script.py"
    print(f"Running {script}...\n")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print("----- STDOUT -----")
    print(result.stdout)
    print("----- STDERR -----")
    print(result.stderr)
    if result.returncode == 0:
        print("\nTest PASSED: Script ran successfully.")
    else:
        print(f"\nTest FAILED: Script exited with code {result.returncode}")
        exit(result.returncode)

if __name__ == "__main__":
    main()

