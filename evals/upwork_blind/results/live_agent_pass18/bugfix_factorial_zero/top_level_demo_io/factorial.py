from pathlib import Path


def factorial(n: int) -> int:
    result = 1
    for value in range(1, n + 1):
        result *= value
    return result


_demo_path = Path(__file__).resolve().parent / "data" / "example.csv"
print("factorial.py demo")
if _demo_path.exists():
    print(f"reading sample inputs from {_demo_path}")
    for raw_line in _demo_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not raw_line.strip():
            continue
        n = int(raw_line.strip())
        print(f"factorial({n}) = {factorial(n)}")
else:
    print(f"sample file not found: {_demo_path}")
    for n in (0, 1, 5, 7):
        print(f"factorial({n}) = {factorial(n)}")
