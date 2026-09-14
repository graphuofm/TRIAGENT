#!/usr/bin/env bash
# Upload hf_dataset/ to the Hugging Face Hub as a public dataset.
#
# Needs a Hugging Face access token with WRITE permission:
#   https://huggingface.co/settings/tokens  ->  Create new token  ->  Type: Write
#
# The token is read interactively, passed to Python through the environment
# (never on the command line), and unset before exit.
#
# Usage:
#   bash scripts/upload_hf_dataset.sh                      # -> dingjiacheng/triagent
#   bash scripts/upload_hf_dataset.sh someuser/some-name   # custom repo id

set -uo pipefail

REPO_ID="${1:-dingjiacheng/triagent}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${ROOT}/hf_dataset"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }

bold "0. Checking the built dataset"
for f in README.md data/fpb.parquet data/tfns.parquet data/fpb_zh.parquet data/fpb_adversarial.parquet data/scd_index.parquet; do
  if [ -f "${DIR}/${f}" ]; then ok "${f}"; else bad "missing ${f} — run: python3 scripts/build_hf_dataset.py"; exit 1; fi
done

printf '\nHugging Face WRITE token (input hidden): '
read -rs HF_TOKEN_INPUT
printf '\n\n'
if [ -z "${HF_TOKEN_INPUT}" ]; then bad "No token entered."; exit 1; fi

HF_TOKEN="${HF_TOKEN_INPUT}" REPO_ID="${REPO_ID}" DIR="${DIR}" python3 - <<'PY'
import os, sys
from huggingface_hub import HfApi

token, repo_id, folder = os.environ["HF_TOKEN"], os.environ["REPO_ID"], os.environ["DIR"]
api = HfApi(token=token)

def ok(m):  print(f"  \033[32m✓\033[0m {m}")
def bad(m): print(f"  \033[31m✗\033[0m {m}")

print("\033[1m1. Verifying token\033[0m")
try:
    who = api.whoami()
    ok(f"logged in as {who['name']}")
except Exception as e:
    bad(f"token rejected: {e}"); sys.exit(1)

owner = repo_id.split("/")[0]
if owner != who["name"] and owner not in [o["name"] for o in who.get("orgs", [])]:
    bad(f"token belongs to {who['name']}, cannot write to {owner}/"); sys.exit(1)

print("\n\033[1m2. Creating the dataset repo\033[0m")
try:
    url = api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    ok(f"{url}")
except Exception as e:
    bad(f"create_repo failed: {e}"); sys.exit(1)

print("\n\033[1m3. Uploading files\033[0m")
try:
    info = api.upload_folder(
        folder_path=folder, repo_id=repo_id, repo_type="dataset",
        commit_message="TriAgent committee predictions (CIKM 2026, arXiv:2607.19794)",
    )
    ok(f"commit {info.oid[:10]}")
except Exception as e:
    bad(f"upload failed: {e}"); sys.exit(1)

print("\n\033[1m4. Verifying\033[0m")
files = sorted(api.list_repo_files(repo_id, repo_type="dataset"))
ok(f"{len(files)} files on the Hub")
for f in files:
    if not f.startswith("summaries/"):
        print(f"      {f}")
print(f"      summaries/  ({sum(f.startswith('summaries/') for f in files)} CSV)")
print(f"\n  https://huggingface.co/datasets/{repo_id}")
PY
status=$?
unset HF_TOKEN_INPUT
exit $status
