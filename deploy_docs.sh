#!/bin/bash
set -e

# Navigate to docs directory
cd R15BSI/docs

# Build docs
py -m sphinx -b html source build

# Ensure .nojekyll exists to prevent GitHub from ignoring _* files
touch build/html/.nojekyll

# Back to repo root
cd ../..

# Prepare worktree
git worktree add /tmp/gh-pages gh-pages
rm -rf /tmp/gh-pages/*

# Copy complete contents of built docs
cp -a R15BSI/docs/build/html/. /tmp/gh-pages/

# Commit and push
cd /tmp/gh-pages
git add --all
git commit -m "Update docs $(date +'%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
git push origin gh-pages

# Cleanup
cd -
git worktree remove /tmp/gh-pages
