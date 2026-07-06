#!/usr/bin/env bash

set -euo pipefail

cd test

# Remove treetime_examples in case it exists to not fail
rm -rf treetime_examples
git clone https://github.com/neherlab/treetime_examples.git

bash command_line_tests.sh

pytest test_treetime.py
pytest test_vcf.py
pytest test_site_rate_model.py
pytest test_site_rate_cli.py

# Clean up, the 202* is to remove auto-generated output dirs
rm -rf treetime_examples __pycache__ 202*
