#!/bin/bash
# rebuild.sh: pull r2-time, build, restart preview (detached; poll /tmp/build.log for BUILD_EXIT)
cd /workspace/home/cahidarda.github.io
git fetch -q origin r2-time && git checkout -q -B r2-time FETCH_HEAD && git log --oneline -1
export npm_config_manage_package_manager_versions=false
pkill -f "astro preview" 2>/dev/null
setsid bash -c 'pnpm build > /tmp/build.log 2>&1; echo BUILD_EXIT=$? >> /tmp/build.log; setsid nohup pnpm preview --host 0.0.0.0 --port 4321 > /tmp/preview.log 2>&1 &' > /dev/null 2>&1 < /dev/null &
echo started
