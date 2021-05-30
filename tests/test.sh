#!/bin/sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ts-node "$SCRIPT_DIR/utils/clusterTools/testscript.ts"

mocha --recursive 'tests/**/*.spec.ts'