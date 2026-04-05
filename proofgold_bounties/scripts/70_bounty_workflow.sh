#!/bin/bash
# 70_bounty_workflow.sh - Interactive bounty workflow helper
# Fetch bounties, create drafts, validate locally

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

LAVA_BINARY="$(pwd)/scripts/proofgoldlava"
DRAFTS_DIR="$(pwd)/drafts"
TEMPLATES_DIR="$(pwd)/drafts/templates"

echo "=========================================="
echo "Proofgold Bounty Workflow"
echo "=========================================="
echo

# Check if Proofgold is running
if ! pgrep -f "proofgoldlava" >/dev/null; then
    echo -e "${RED}ERROR: Proofgold Lava is not running${NC}"
    echo "Start it first: ./scripts/50_start_stack.sh"
    exit 1
fi

# Menu
echo "Select an action:"
echo "  1) List available bounties"
echo "  2) Create new draft from template"
echo "  3) Validate existing draft (readdraft)"
echo "  4) Show bounty statistics"
echo "  5) Help: Proofgold document format"
echo
read -p "Choice [1-5]: " ACTION

case $ACTION in
    1)
        # List bounties
        echo
        echo "Fetching available bounties..."
        echo "================================"
        echo

        # This depends on the actual Lava interface
        # Common approaches:
        # - RPC call to list bounties
        # - Command line query
        # - Query from local database

        echo "Attempting to query bounties via Proofgold..."

        # Try different potential commands
        if timeout 10 $LAVA_BINARY listbounties &>/dev/null; then
            $LAVA_BINARY listbounties
        elif timeout 10 $LAVA_BINARY bounties &>/dev/null; then
            $LAVA_BINARY bounties
        else
            echo -e "${YELLOW}Direct query not available${NC}"
            echo
            echo "Alternative methods:"
            echo "  1) Use Proofgold explorer: https://proofgold.org"
            echo "  2) Query via Proofgold console (if available)"
            echo "  3) Check recent blocks for bounty transactions"
            echo
            echo "Common bounty locations:"
            echo "  - HOHF reward bounties (legacy theory)"
            echo "  - Custom theories with prize pools"
        fi

        echo
        echo "For HOHF reward bounties, see:"
        echo "  https://prfgld.github.io/publishing.html"
        ;;

    2)
        # Create new draft
        echo
        echo "Create new draft"
        echo "================"
        echo

        # Ask for theory ID
        echo "Common theory IDs:"
        echo "  - HOHF (Higher Order Higher Form - has reward bounties)"
        echo "  - Custom theory ID from blockchain"
        echo
        read -p "Enter theory ID (default: HOHF): " THEORY_ID
        THEORY_ID=${THEORY_ID:-HOHF}

        # Ask for proposition
        echo
        echo "Enter target proposition:"
        echo "(This should be a proposition hash from the bounty list)"
        read -p "Proposition ID: " PROP_ID

        if [ -z "$PROP_ID" ]; then
            echo -e "${RED}ERROR: Proposition ID required${NC}"
            exit 1
        fi

        # Generate draft filename
        DRAFT_FILE="$DRAFTS_DIR/draft_${THEORY_ID}_${PROP_ID}_$(date +%Y%m%d_%H%M%S).pfg"

        echo
        echo "Creating draft: $DRAFT_FILE"

        # Check if template exists
        TEMPLATE_FILE="$TEMPLATES_DIR/hohf_document.template"

        if [ -f "$TEMPLATE_FILE" ]; then
            echo "Using template: $TEMPLATE_FILE"
            cp "$TEMPLATE_FILE" "$DRAFT_FILE"

            # Replace placeholders
            sed -i "s/THEORY_ID_PLACEHOLDER/$THEORY_ID/g" "$DRAFT_FILE"
            sed -i "s/PROPOSITION_ID_PLACEHOLDER/$PROP_ID/g" "$DRAFT_FILE"
        else
            # Create from scratch
            cat > "$DRAFT_FILE" <<EOF
Document $THEORY_ID

(* Proofgold Draft Document *)
(* Theory: $THEORY_ID *)
(* Target Proposition: $PROP_ID *)
(* Created: $(date) *)

(* ====================================== *)
(* Known declarations (imports) *)
(* ====================================== *)

(* Example: Import base logic *)
(* Known and : prop -> prop -> prop *)

(* ====================================== *)
(* Definitions *)
(* ====================================== *)

(* Define needed concepts here *)
(* Example:
Def myConcept : set -> prop :=
  fun x:set => ...
*)

(* ====================================== *)
(* Theorem *)
(* ====================================== *)

(* Main theorem proving the target proposition *)
Thm target_theorem : REPLACE_WITH_PROPOSITION_STATEMENT
Proof:
  (* Proof term goes here *)
  (* Example structure:
  PrAp (Lam ... ) ...
  *)
  REPLACE_WITH_PROOF_TERM
Qed.

(* ====================================== *)
(* End of document *)
(* ====================================== *)
EOF
        fi

        echo -e "${GREEN}Draft created!${NC}"
        echo
        echo "Edit the draft:"
        echo "  nano $DRAFT_FILE"
        echo "  vim $DRAFT_FILE"
        echo
        echo "When ready, validate with option 3"
        ;;

    3)
        # Validate draft
        echo
        echo "Validate draft (readdraft)"
        echo "=========================="
        echo

        # List existing drafts
        echo "Available drafts:"
        DRAFTS=($(ls -1 "$DRAFTS_DIR"/*.pfg 2>/dev/null || echo ""))

        if [ ${#DRAFTS[@]} -eq 0 ]; then
            echo -e "${YELLOW}No drafts found in $DRAFTS_DIR${NC}"
            echo "Create one with option 2"
            exit 0
        fi

        i=1
        for draft in "${DRAFTS[@]}"; do
            echo "  $i) $(basename $draft)"
            ((i++))
        done

        echo
        read -p "Select draft number (or 'q' to quit): " DRAFT_NUM

        if [ "$DRAFT_NUM" = "q" ]; then
            exit 0
        fi

        SELECTED_DRAFT="${DRAFTS[$((DRAFT_NUM-1))]}"

        if [ ! -f "$SELECTED_DRAFT" ]; then
            echo -e "${RED}Invalid selection${NC}"
            exit 1
        fi

        echo
        echo "Validating: $(basename $SELECTED_DRAFT)"
        echo "----------------------------------------"
        echo

        # Run readdraft (this depends on actual Lava interface)
        if timeout 30 $LAVA_BINARY readdraft "$SELECTED_DRAFT" 2>&1; then
            echo
            echo -e "${GREEN}✓ Validation PASSED${NC}"
            echo
            echo "Draft is syntactically correct!"
            echo
            echo "Next steps:"
            echo "  ./scripts/80_publish.sh (commit and publish)"
        else
            echo
            echo -e "${RED}✗ Validation FAILED${NC}"
            echo
            echo "Fix errors in the draft and try again"
            echo "  Edit: nano $SELECTED_DRAFT"
        fi
        ;;

    4)
        # Bounty statistics
        echo
        echo "Bounty Statistics"
        echo "================="
        echo

        echo "Attempting to gather bounty stats..."

        # This would ideally query the blockchain for:
        # - Total bounties available
        # - Total value locked
        # - Recently claimed bounties
        # - Your current holdings

        # Placeholder - depends on actual Lava interface
        if timeout 10 $LAVA_BINARY stats &>/dev/null; then
            $LAVA_BINARY stats
        elif timeout 10 $LAVA_BINARY printassets &>/dev/null; then
            echo "Your assets:"
            $LAVA_BINARY printassets
        else
            echo -e "${YELLOW}Stats query not available via CLI${NC}"
            echo
            echo "Use Proofgold explorer for statistics:"
            echo "  https://proofgold.org"
        fi
        ;;

    5)
        # Help
        echo
        echo "Proofgold Document Format"
        echo "========================="
        echo
        cat <<'EOF'
A Proofgold document has this structure:

1. Document Header:
   Document THEORY_ID

2. Known Declarations (imports):
   Known name : type

   Example:
   Known and : prop -> prop -> prop

3. Definitions:
   Def name : type := definition

   Example:
   Def myFunc : set -> set := fun x => x

4. Theorems:
   Thm name : statement
   Proof:
     proof_term
   Qed.

5. Proof Terms:
   Proofgold uses an "assembly language" for proofs:
   - Lam : lambda abstraction
   - Ap : application
   - PrAp : proof application
   - All : forall quantifier
   - Imp : implication

   Example proof term:
   Lam (fun H1:prop => Lam (fun H2:prop => PrAp H1 H2))

Resources:
- Grammar: https://prfgld.github.io/ihol.html
- Publishing: https://prfgld.github.io/publishing.html
- Examples: Search Proofgold blockchain for published documents

For automated proving:
- Consider using HOL4 + export to Proofgold format
- Or manual construction following the grammar
EOF
        ;;

    *)
        echo -e "${RED}Invalid selection${NC}"
        exit 1
        ;;
esac

echo
echo "=========================================="
