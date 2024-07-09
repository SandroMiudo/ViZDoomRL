#! /bin/bash

readonly CONDA_ENV="vizdoom_env"
readonly ROOT_DIR="ViZDoomRL"
readonly REQUIRED_BASH_VERSION=4 

set -e

bash_version=$(bash --version | grep -E ".+([0-9]+\.){2}[0-9]+" | tr -d "[:alpha:], \t" | cut -d "." -f1) 

if [ "${bash_version}" -lt "${REQUIRED_BASH_VERSION}" ]; then
  printf "Requires bash version >= ${REQUIRED_BASH_VERSION}.0\n"; exit 1;
fi

printf "Bash version used = %d\n" "${bash_version}"

if [[ $(pwd) != */"${ROOT_DIR}" ]]; then 
  printf "script needs to be invoked from root dir\n"; exit 1;
fi

if [[ $(conda env list | cut -d" " -f1 | grep "${CONDA_ENV}") != "${CONDA_ENV}" ]]; then
  printf "create new conda environment\n"
  conda env create --no-default-packages -f conda_env.yml > /dev/null
fi

conda activate "${CONDA_ENV}"
printf "activate enviroment\n"

# Just two simple default invocations (using default arguments)

if [ "$1" = "DBG" ]; then
  printf "executing with debug information\n"
  python3 -m src.agent.RLSingleQNetwork --train --debug --resX 320 --resY 240
else 
  python3 -m src.agent.RLSingleQNetwork --train --resX 320 --resY 240
fi