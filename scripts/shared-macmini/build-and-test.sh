#!/bin/bash

# Build and test Serac on team's shared MacMini, then report results to a set of emails

# Variables
CI_ROOT_DIR="/Users/chapman39/dev/serac/ci"
PROJECT_DIR="$CI_ROOT_DIR/repo"
OUTPUT_LOG="$CI_ROOT_DIR/logs/macmini-build-and-test-$(date +"%Y_%m_%d_%H_%M_%S").log"
HOST_CONFIG="$CI_ROOT_DIR/host-configs/firion-darwin-sonoma-aarch64-clang@14.0.6.cmake"
RECIPIENTS="chapman39@llnl.gov,white238@llnl.gov,talamini1@llnl.gov"

# Go to project directory
cd $PROJECT_DIR

# Update Serac
git checkout task/chapman39/macmini-build >> $OUTPUT_LOG 2>&1 # TODO CHANGE TO DEVELOP
git pull >> $OUTPUT_LOG 2>&1
git submodule update --init --recursive >> $OUTPUT_LOG 2>&1

# Clear previous build(s)
rm -rfv _serac_build_and_test* >> $OUTPUT_LOG 2>&1

# Build and test Serac
./scripts/llnl/build_src.py --host-config $HOST_CONFIG -v -j16 >> $OUTPUT_LOG 2>&1

# Email variables
if [ $? -eq 0 ]; then
    EMAIL_SUBJECT="Serac Succeeded!"
else
    EMAIL_SUBJECT="Serac Failed!"
fi
EMAIL_SUBJECT="$EMAIL_SUBJECT MacMini build and test report $(date)"
EMAIL_BODY="This is automatic weekly report of Serac's MacMini build. See attached for log."

# Send report via email
echo "$EMAIL_BODY" | mutt -a "$OUTPUT_LOG" -s "$EMAIL_SUBJECT" -- "$RECIPIENTS"
