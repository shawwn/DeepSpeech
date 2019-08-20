#!/bin/bash

# https://stackoverflow.com/questions/59895/get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

#exec python3 "$DIR/deepspeech.py" --model "$DIR/models/output_graph.pbmm" --alphabet "$DIR/models/alphabet.txt" --audio "$@"
exec deepspeech --model "$DIR/models/output_graph.pbmm" --alphabet "$DIR/models/alphabet.txt" --audio "$@"
