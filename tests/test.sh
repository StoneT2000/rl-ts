#!/bin/sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

nyc --reporter=json mocha -r ts-node/register -r tsconfig-paths/register --recursive 'tests/**/*.spec.ts' && ts-node "tests/utils/clusterTools/testscript.ts"