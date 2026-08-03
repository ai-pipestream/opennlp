#!/usr/bin/env bash
# setup-workspace.sh — recreate the research workspace from this preview build.
#
# This branch (kristian-3.x-features) is an integration line: apache main plus
# every contributing research branch, recorded ref-by-ref in
# PIPESTREAM-PROVENANCE.txt. This script turns that manifest back into a
# working tree per branch, so you can inspect or build any feature in
# isolation instead of archaeology on the merged result.
#
#   bash setup-workspace.sh [workspace-dir]
#
#   workspace-dir  where the worktrees go; defaults to the parent directory
#                  of this checkout (the krick layout: /work/worktrees/opennlp
#                  contains uber/ plus one directory per branch).
#
# Options:
#   --pin      detach each worktree at the exact sha recorded in the
#              provenance manifest instead of the branch tip (reproduces this
#              build precisely, even if branches have moved on).
#   --dry-run  print what would be created and exit.
#
# Needs: git 2.30+, network access to github.com/ai-pipestream/opennlp.
# Safe to re-run: existing directories are left untouched.

set -euo pipefail

pin=0 dry=0 ws=""
for arg in "$@"; do
  case "$arg" in
    --pin) pin=1 ;;
    --dry-run) dry=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) ws="$arg" ;;
  esac
done

here="$(cd "$(dirname "$0")" && pwd)"
manifest="$here/PIPESTREAM-PROVENANCE.txt"
[ -f "$manifest" ] || { echo "error: PIPESTREAM-PROVENANCE.txt not found next to this script" >&2; exit 1; }

cd "$here"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "error: run this from a git checkout of the preview branch" >&2; exit 1; }

# Find (or add) the remote pointing at the ai-pipestream fork, preferring
# the GitHub one (development home) over any mirror.
remote=""
for r in $(git remote); do
  case "$(git remote get-url "$r")" in
    *github.com*ai-pipestream/opennlp*) remote="$r"; break ;;
  esac
done
if [ -z "$remote" ]; then
  for r in $(git remote); do
    case "$(git remote get-url "$r")" in
      *ai-pipestream/opennlp*) remote="$r"; break ;;
    esac
  done
fi
if [ -z "$remote" ]; then
  git remote add github https://github.com/ai-pipestream/opennlp.git
  remote=github
fi

ws="${ws:-$(dirname "$here")}"
mkdir -p "$ws"
echo "workspace: $ws   remote: $remote   mode: $([ $pin = 1 ] && echo pinned-shas || echo branch-tips)"

[ $dry = 1 ] || git fetch --quiet "$remote"

added=0 kept=0 moved=0
while read -r _ sha branch; do
  dest="$ws/$branch"
  if [ -e "$dest" ]; then
    kept=$((kept+1)); continue
  fi
  if [ $dry = 1 ]; then
    echo "would add  $branch"
    added=$((added+1)); continue
  fi
  if [ $pin = 1 ]; then
    git worktree add --quiet --detach "$dest" "$sha"
    echo "added   $branch  (pinned $sha)"
  else
    git worktree add --quiet --track -b "$branch" "$dest" "$remote/$branch" 2>/dev/null \
      || git worktree add --quiet "$dest" "$branch"
    tip="$(git -C "$dest" rev-parse --short=9 HEAD)"
    if [ "$tip" != "$sha" ]; then
      echo "added   $branch  (tip $tip; build used $sha — branch moved since, use --pin to reproduce)"
      moved=$((moved+1))
    else
      echo "added   $branch"
    fi
  fi
  added=$((added+1))
done < <(grep '^ref: ' "$manifest")

echo "done: $added added, $kept already present$([ $moved -gt 0 ] && echo ", $moved branch(es) ahead of this build")"
echo "note: build each branch with its own ./mvnw; install opennlp-api from a"
echo "      branch's worktree before building that branch's extension modules."
