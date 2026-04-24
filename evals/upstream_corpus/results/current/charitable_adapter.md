# Pass-7-b Charitable Adapter Report

## Provenance
- **tlaplus/Examples** @ `d9ce4db7c770cc82e662870bce168ff8f59aff24`

## Measurement
- Baseline Pass-7 intrinsic `revise` cases: **44**
- Flipped `revise` → `accept` after charitable extraction: **21**
- Remaining `revise` after extraction: **23**
- Regressed (revise → reject): **0**
- Extraction-debt share: 47.7% of the revise load was extraction debt (comments already in the .tla file)

## Extraction rules (strict)
- Only extracts `(* block comments *)` and `\* line comments` already present in .tla files
- Preserves declaration lines (VARIABLES, action definitions) that carry inline comments, so variable/action names travel with their explanations
- Writes the extracted text to `extracted_tla_comments.md` inside a tempdir bundle copy
- Does NOT modify upstream repos; does NOT synthesize evidence; does NOT paraphrase

## Cases flipped `revise` → `accept`
- `Bakery-Boulangerie`
- `LeastCircularSubstring`
- `LoopInvariance`
- `Majority`
- `MultiPaxos-SMR`
- `PaxosHowToWinATuringAward`
- `SimplifiedFastPaxos`
- `SlidingPuzzles`
- `SpanningTree`
- `YoYo`
- `acp`
- `barriers`
- `bcastByz`
- `bcastFolklore`
- `braf`
- `byihive`
- `byzpaxos`
- `dag-consensus`
- `detector_chan96`
- `ewd840`
- `nbacc_ray97`

## Cases still `revise` after extraction
- `Chameneos`
- `CheckpointCoordination`
- `CigaretteSmokers`
- `Disruptor`
- `FiniteMonotonic`
- `GameOfLife`
- `KeyValueStore`
- `KnuthYao`
- `SDP_Attack_New_Solution_Spec`
- `SDP_Attack_Spec`
- `TeachingConcurrency`
- `aba-asyn-byz`
- `bosco`
- `btree`
- `c1cs`
- `cbc_max`
- `cf1s-folklore`
- `chang_roberts`
- `diskpaxos`
- `ewd687a`
- `locks_auxiliary_vars`
- `nbacg_guer01`
- `spanning`

## Top remaining revise reasons (after extraction)
- `README does not map action names into outsider-facing prose` — 12
- `README does not explain all state variables: FwState, SDPSvrState, aState, sState` — 2
- `README does not explain all state variables: numMeetings` — 1
- `README does not explain all state variables: cLogs` — 1
- `README does not explain all state variables: grid` — 1
- `README does not explain all state variables: nSntE` — 1
- `README does not explain all state variables: pc` — 1
- `README does not explain all state variables: root` — 1
- `README does not explain all state variables: bcastMsg, dValue, pc, rcvdMsg` — 1
- `README does not explain all state variables: dval, nCrash, pc, rcvdMsgs, sntMsgs` — 1
- `README does not explain all state variables: allInput, blocksRead, chosen, dblock, disksWritten, output, phase` — 1
- `README does not explain all state variables: activeSons, terminationDetected` — 1
- `README does not explain all state variables: nSntNo` — 1
- `README does not explain all state variables: msg, prnt, rpt` — 1
