#!/bin/sh
# Clone SDDNet (no licence file in the repository; used for research reproduction only, not redistributed)
# at the commit used for every run in the paper. The code is imported, not modified.
set -eu
cd "$(dirname "$0")"
mkdir -p upstream
[ -d upstream/SDDNet ] || git clone https://github.com/rmcong/SDDNet_ACMMM23.git upstream/SDDNet
git -C upstream/SDDNet checkout dddcdd4673dc24a411addcc9d4cbc63a6b30da73
