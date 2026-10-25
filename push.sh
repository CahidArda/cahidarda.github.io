#!/bin/bash
# push.sh <dir>: commit the shots into a scratch worktree and push to r2-time-scratch
set -e
D=$1; R=/workspace/home/cahidarda.github.io; W=/tmp/scratchwt
cd $R; git fetch -q origin r2-time-scratch
rm -rf $W; git worktree prune; git worktree add -q -f $W FETCH_HEAD
mkdir -p $W/$D && cp ~/pw/$D/* $W/$D/
cd $W && git add $D && git -c user.name=Arda -c user.email=cahidardaooz@hotmail.com commit -qm "shots $D" && git push -q origin HEAD:r2-time-scratch && echo pushed
