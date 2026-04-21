#!/usr/bin/env bash
set -euo pipefail

# Rule 30 Submission Script
# Sends the bounded non-periodicity certificate to Wolfram Research

ROOT="/home/player2/signal_experiments/qa_alphageometry_ptolemy/rule30_submission_package"
TARBALL="/home/player2/signal_experiments/qa_alphageometry_ptolemy/rule30_submission_package.tar.gz"
EMAIL_TXT="$ROOT/rule30_submission_email.txt"

: "${FROM:?set FROM=your@email.com before running (sender address for MIME From header)}"
to="contact@wolframscience.com"
subject="Bounded Non-Periodicity Certificate for Rule 30 Center Column"

# Verify files exist
if [[ ! -f "$TARBALL" ]]; then
    echo "ERROR: Tarball not found at $TARBALL"
    exit 1
fi

if [[ ! -f "$EMAIL_TXT" ]]; then
    echo "ERROR: Email text not found at $EMAIL_TXT"
    exit 1
fi

echo "Sending Rule 30 submission to: $to"
echo "Attachment: $TARBALL ($(du -h "$TARBALL" | cut -f1))"

# Build MIME email with attachment
boundary="----=_rule30_boundary_$(date +%s)"

{
  echo "To: $to"
  echo "From: $FROM"
  echo "Subject: $subject"
  echo "MIME-Version: 1.0"
  echo "Content-Type: multipart/mixed; boundary=\"$boundary\""
  echo
  echo "--$boundary"
  echo "Content-Type: text/plain; charset=\"utf-8\""
  echo "Content-Transfer-Encoding: 7bit"
  echo
  cat "$EMAIL_TXT"
  echo
  echo "--$boundary"
  echo "Content-Type: application/gzip; name=\"rule30_submission_package.tar.gz\""
  echo "Content-Transfer-Encoding: base64"
  echo "Content-Disposition: attachment; filename=\"rule30_submission_package.tar.gz\""
  echo
  base64 "$TARBALL"
  echo
  echo "--$boundary--"
} | msmtp -t

echo ""
echo "✔ Rule 30 submission sent successfully!"
echo "  To: $to"
echo "  Subject: $subject"
echo "  Attachment: rule30_submission_package.tar.gz"
echo ""
echo "Check ~/.msmtp.log for delivery confirmation."
