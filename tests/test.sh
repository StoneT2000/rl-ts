#!/bin/sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mocha --recursive 'tests/**/*.spec.ts'

ts-node "tests/utils/clusterTools/testscript.ts"