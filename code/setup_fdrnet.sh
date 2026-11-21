#!/bin/sh
# Clone FDRNet (MIT licence) at the commit used for every run in the paper. The code is imported, not modified.
set -eu
cd "$(dirname "$0")"
mkdir -p upstream
[ -d upstream/FDRNet ] || git clone https://github.com/rayleizhu/FDRNet.git upstream/FDRNet
git -C upstream/FDRNet checkout ea7e14730896225d12caaf4a4fb8c6ec3ec040ef
