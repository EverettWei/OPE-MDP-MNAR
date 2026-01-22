import sys
from scripts.simulation import main as sim_main

def main():
    if len(sys.argv) == 1:
        # default args when called without CLI
        sys.argv = [
            "scripts.simulation",
            "--n", "500", "--T", "10", "--gamma", "1.0",
            "--seed", "42", "--device", "cpu"
        ]
    sim_main()

if __name__ == "__main__":
    main()