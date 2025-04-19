git init

find . -type f ! -path '*/\.git/*' | while IFS= read -r file; do
    if [[ "$OSTYPE" == "darwin"* ]]; then
        mod_time=$(stat -f "%Sm" -t "%Y-%m-%dT%H:%M:%S" "$file")
    else
        mod_time=$(date -r "$file" "+%Y-%m-%dT%H:%M:%S")
    fi
    git add "$file"
    GIT_COMMITTER_DATE="$mod_time" git commit --date="$mod_time" -m "Add $file"
done
