#!/bin/bash
set -e

# Build docs
py -m sphinx -b html source build

# Switch to gh-pages branch (or clone a fresh copy)
git worktree add /tmp/gh-pages gh-pages

# Remove old files
rm -rf /tmp/gh-pages/*

# Copy new docs
cp -r build/* /tmp/gh-pages/

# Commit and push
cd /tmp/gh-pages
git add --all
git commit -m "Update docs $(date +'%Y-%m-%d %H:%M:%S')"
git push origin gh-pages

# Cleanup worktree
cd -
git worktree remove /tmp/gh-pages