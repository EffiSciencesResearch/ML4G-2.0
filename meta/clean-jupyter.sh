#!/bin/bash

set -e

for file in "$@"; do
    # Based on https://timstaley.co.uk/posts/making-git-and-jupyter-notebooks-play-nice/#id2
    out=$(jq --indent 1 \
        '
        (.cells[] | select(has("outputs")) | .outputs) = []
        | (.cells[] | select(has("execution_count")) | .execution_count) = null
        | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}}
        | .cells[].metadata = {}
        ' "$file")
    echo "$out" > "$file"
done
