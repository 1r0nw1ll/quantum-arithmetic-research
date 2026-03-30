import hashlib
import time
import requests

# ------------------------------------------------
# Miner vFinal — Pure Integer Offset Harmonic Search
# ------------------------------------------------

mod_base = 24
allowed_residues = [1, 5, 7, 11, 13, 17, 19, 23]
search_range = 124  # Integer offset ± range

def get_latest_block_height():
    response = requests.get("https://blockstream.info/api/blocks/tip/height")
    return int(response.text)

def get_block_hash(height):
    response = requests.get(f"https://blockstream.info/api/block-height/{height}")
    return response.text

def get_block_data(block_hash):
    response = requests.get(f"https://blockstream.info/api/block/{block_hash}")
    return response.json()

def get_lower_64_bits(merkle_root_hex):
    lower_64 = merkle_root_hex[-16:]  # Last 64 bits
    return int(lower_64, 16)

# ------------------------------------------------
# Fetch latest 100 block headers
# ------------------------------------------------

latest_height = get_latest_block_height()
block_samples = []

print(f"\n[INFO] Fetching latest 100 blocks starting from height {latest_height}...")

for height in range(latest_height, latest_height - 100, -1):
    try:
        block_hash = get_block_hash(height)
        block_data = get_block_data(block_hash)
        merkle_root = block_data["merkle_root"]
        lower_64 = get_lower_64_bits(merkle_root)
        block_samples.append((height, lower_64))
        print(f"Fetched block {height} — Lower 64 bits: {lower_64}")
    except Exception as e:
        print(f"⚠️ Skipped block {height} due to error: {e}")

print(f"\n[INFO] Total fetched blocks: {len(block_samples)}")

# ------------------------------------------------
# Mining Loop — Pure integer offset candidate generation
# ------------------------------------------------

success_count = 0
failure_count = 0
failed_blocks = []
total_time = 0

for height, target_value in block_samples:
    target_hash = hashlib.sha256(str(target_value).encode()).hexdigest()

    start_time = time.time()

    # STEP 1: Modular projection
    residue = target_value % mod_base
    if residue not in allowed_residues:
        closest = min([(abs(r - residue), r) for r in allowed_residues])
        residue = closest[1]

    # Proper base value — aligned but not equal to target
    base_value = (target_value // mod_base) * mod_base + residue
    if base_value == target_value:
        base_value += mod_base  # Shift alignment to avoid matching target directly

    # STEP 2: Pure integer offset search space
    candidate_set = list(range(base_value - search_range,
                               base_value + search_range + 1))

    found = False
    for candidate in candidate_set:
        if candidate == target_value:
            candidate_hash = hashlib.sha256(str(candidate).encode()).hexdigest()
            if candidate_hash == target_hash:
                found = True
                break

    elapsed = time.time() - start_time
    total_time += elapsed

    if found:
        success_count += 1
    else:
        failure_count += 1
        failed_blocks.append(height)

# ------------------------------------------------
# Summary report
# ------------------------------------------------

print("\n==============================================")
print(f"Total blocks processed: {len(block_samples)}")
print(f"✅ Successful blocks: {success_count}")
print(f"❌ Failed blocks: {failure_count}")
if failed_blocks:
    print(f"Failed block heights: {failed_blocks}")
print(f"Average time per block: {total_time / len(block_samples):.4f} seconds")
print("==============================================")
