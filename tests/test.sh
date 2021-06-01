#!/bin/sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mocha -r ts-node/register -r tsconfig-paths/register --recursive 'tests/**/*.spec.ts'

ts-node "tests/utils/clusterTools/testscript.ts"