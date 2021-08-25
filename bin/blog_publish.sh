#!/bin/bash

# Args:
#   1 (str): notebook file basename, ie, without the .extension
# Example:
#   bash bin/blog_publish.sh Bayesian_exponential_model 2021-08-04-blog-post_predicting_nsw_covid_cases

project_root=$(git rev-parse --show-toplevel)

cd ${project_root}/notebooks

echo "In folder $(pwd)"

# jupyter nbconvert --to html $1.ipynb
cat <<EOF > $1.html
$(cat $1.jekyll)

$(cat $1.html)
EOF

mv $1.html $2.html