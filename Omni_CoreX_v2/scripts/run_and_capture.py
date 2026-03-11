import subprocess
import sys

try:
    result = subprocess.run([sys.executable, 'test_graph.py'], capture_output=True, text=True)
    with open('test_output.txt', 'w', encoding='utf-8') as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    print("Execution finished. Check test_output.txt")
except Exception as e:
    print(f"Failed to run: {e}")
