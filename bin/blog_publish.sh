#!/bin/bash

# Args:
#   1 (str): notebook file basename, ie, without the .extension
# Example:
#   $ bash export_html.sh Bayesian_exponential_model

jupyter nbconvert --to html $1.ipynb
cat <<EOF > $1.html
$(cat $1.jekyll)

$(cat $1.html)
EOF