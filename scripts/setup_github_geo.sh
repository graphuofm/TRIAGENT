#!/usr/bin/env bash
# Set GitHub repo metadata + enable Pages for TriAgent (GEO setup).
#
# Needs a GitHub Personal Access Token:
#   Classic PAT  -> scope: repo
#   Fine-grained -> Repository permissions: Administration (write), Pages (write)
#   Create at: https://github.com/settings/tokens
#
# The token is read interactively so it never lands in your shell history.
# Safe to re-run; every step is idempotent.
#
# Usage:  bash scripts/setup_github_geo.sh

set -uo pipefail

OWNER="graphuofm"
REPO="TRIAGENT"
API="https://api.github.com/repos/${OWNER}/${REPO}"
SITE="https://${OWNER}.github.io/${REPO}/"

DESCRIPTION="Divergence-aware multi-agent routing that cuts LLM inference cost 10-100x. One signal routes queries, keys a multilingual cache, and detects hallucinations at AUC 0.90. CIKM 2026."

TOPICS='["llm","llm-inference","llm-routing","inference-optimization","model-cascade","semantic-cache","multi-agent","hallucination-detection","cost-optimization","financial-sentiment-analysis","finbert","nlp","cikm2026","llm-serving"]'

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }

# ---------------------------------------------------------------- token
printf 'GitHub PAT (input hidden): '
read -rs GH_TOKEN
printf '\n\n'

if [ -z "${GH_TOKEN}" ]; then
  bad "No token entered. Aborting."
  exit 1
fi

auth=(-H "Authorization: Bearer ${GH_TOKEN}" -H "Accept: application/vnd.github+json")

# ---------------------------------------------------------------- 0. verify
bold "0. Verifying token and repo access"
code=$(curl -sS -o /tmp/gh_check.json -w '%{http_code}' "${auth[@]}" "${API}")
if [ "$code" != "200" ]; then
  bad "Cannot read ${OWNER}/${REPO} (HTTP ${code})."
  python3 -c "import json;print('   ',json.load(open('/tmp/gh_check.json')).get('message','?'))" 2>/dev/null
  bad "Check that the token is valid and has repo access."
  exit 1
fi
ok "Token works, repo reachable"

# ---------------------------------------------------------------- 1. metadata
bold ""
bold "1. Setting description, website, and feature flags"
code=$(curl -sS -o /tmp/gh_patch.json -w '%{http_code}' -X PATCH "${auth[@]}" "${API}" \
  -d "$(python3 -c "
import json,sys
print(json.dumps({
  'description': sys.argv[1],
  'homepage': sys.argv[2],
  'has_issues': True,
  'has_wiki': False,
  'has_projects': False,
}))" "$DESCRIPTION" "$SITE")")

if [ "$code" = "200" ]; then
  ok "Description set"
  ok "Website set to ${SITE}"
  ok "Issues on, wiki/projects off"
else
  bad "Failed (HTTP ${code})"
  python3 -c "import json;print('   ',json.load(open('/tmp/gh_patch.json')).get('message','?'))" 2>/dev/null
fi

# ---------------------------------------------------------------- 2. topics
bold ""
bold "2. Setting topics"
code=$(curl -sS -o /tmp/gh_topics.json -w '%{http_code}' -X PUT "${auth[@]}" \
  "${API}/topics" -d "{\"names\":${TOPICS}}")

if [ "$code" = "200" ]; then
  n=$(python3 -c "import json;print(len(json.load(open('/tmp/gh_topics.json'))['names']))")
  ok "${n} topics set"
else
  bad "Failed (HTTP ${code})"
  python3 -c "import json;print('   ',json.load(open('/tmp/gh_topics.json')).get('message','?'))" 2>/dev/null
fi

# ---------------------------------------------------------------- 3. pages
bold ""
bold "3. Enabling GitHub Pages from /docs on main"
code=$(curl -sS -o /tmp/gh_pages.json -w '%{http_code}' -X POST "${auth[@]}" \
  "${API}/pages" -d '{"source":{"branch":"main","path":"/docs"}}')

case "$code" in
  201) ok "Pages enabled" ;;
  409)
    # Already enabled — update the source instead.
    code2=$(curl -sS -o /tmp/gh_pages2.json -w '%{http_code}' -X PUT "${auth[@]}" \
      "${API}/pages" -d '{"source":{"branch":"main","path":"/docs"}}')
    if [ "$code2" = "204" ]; then
      ok "Pages already on; source confirmed as main:/docs"
    else
      bad "Pages exists but source update failed (HTTP ${code2})"
    fi
    ;;
  *)
    bad "Failed (HTTP ${code})"
    python3 -c "import json;print('   ',json.load(open('/tmp/gh_pages.json')).get('message','?'))" 2>/dev/null
    bad "If this says 'Resource not accessible', the token lacks Pages write permission."
    ;;
esac

unset GH_TOKEN
rm -f /tmp/gh_check.json /tmp/gh_patch.json /tmp/gh_topics.json /tmp/gh_pages.json /tmp/gh_pages2.json

# ---------------------------------------------------------------- 4. verify
bold ""
bold "4. Verifying (public API, no token needed)"
sleep 2
curl -sS "${API}" | python3 -c "
import json,sys
d = json.load(sys.stdin)
print('  description :', (d.get('description') or '<empty>')[:75] + ('...' if len(d.get('description') or '') > 75 else ''))
print('  homepage    :', d.get('homepage') or '<empty>')
t = d.get('topics') or []
print(f'  topics ({len(t):2d})  :', ', '.join(t) if t else '<none>')
"

bold ""
bold "Done."
echo "  Pages takes about a minute to build. Then check:"
echo "    ${SITE}"
echo "  And validate the structured data with:"
echo "    https://search.google.com/test/rich-results?url=${SITE}"
