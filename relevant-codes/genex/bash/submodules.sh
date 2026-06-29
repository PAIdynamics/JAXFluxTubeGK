#!/bin/bash

submods=($(git config -f .gitmodules -l | grep "url"))

echo "searching for submodules";
for i in "${submods[@]}"
do
    echo "  found submodule $i"
done

mode=${1}

if [ -z ${mode} ] || ([ ${mode} != "https" ] && [ ${mode} != "ssh" ]); then
    echo "Error: usage is 'submodules.sh mode'", \
          "where mode is either 'https' or 'ssh'!";
    exit 1
else
    echo "converting to $mode";
fi

for i in "${submods[@]}"
do
    submod_name="$(cut -d'.' -f2 <<<"$i")"

    if [ ${mode} == "https" ]; then
        if [[ $i == *"https"* ]]; then
            echo "  $submod_name is already https, skipping..."
            continue
        fi

        url_ssh="$(cut -d'=' -f2 <<<"$i")"
        url_ssh="$(cut -d'@' -f2 <<<"$url_ssh")"
        url_https=$(echo "$url_ssh" | sed "s/:/\//")
        url_https=$(echo "$url_https" | sed "s/.git//")
        url_https="https://${url_https}"

        git submodule set-url "$submod_name" "$url_https"
    fi

    if [ ${mode} == "ssh" ]; then
        if [[ $i == *"git@"* ]]; then
            echo "  $submod_name is already ssh, skipping..."
            continue
        fi

        url_https="$(cut -d'=' -f2 <<<"$i")"
        url_ssh=$(echo "$url_https" | sed "s/https:\/\//git@/")
        url_ssh=$(echo "$url_ssh" | sed "s/\//:/")
        url_ssh="${url_ssh}.git"

        git submodule set-url "$submod_name" "$url_ssh"
    fi
done

echo "done"
