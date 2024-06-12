#!/bin/bash
##############################################################################
# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

function usage() {
    echo "Usage:          ./build_new_dockerfile.sh <compiler_name> <compiler_full_version> <optional_cuda_version>"
    echo "Example:        ./build_new_dockerfile.sh gcc 13.1.0"
    echo "Example (cuda): ./build_new_dockerfile.sh gcc 13.1.0 12-3"
}

# Must be between two and three args
if [ "$#" -eq 2 ] ; then
    using_cuda=false
elif [ "$#" -eq 3 ] ; then
    using_cuda=true
else
    usage
    exit 1
fi

name=$1
ver=$2
cuda_ver=$3

maj_ver="${ver%%\.*}"

if [[ "$ver" != *"."*"."* ]] ; then
    echo "Error: specify full compiler version"
    usage
    exit 1
fi

if [ $using_cuda = true ] ; then
    cuda_maj_ver="${cuda_ver%-*}"
    tag_name="cuda-${cuda_maj_ver}"
    image="ghcr.io/llnl/radiuss:cuda-${cuda_ver}-ubuntu-22.04"
    spec="%${name}@${ver}+cuda+raja+umpire"
else
    tag_name="${name}-${maj_ver}"
    image="ghcr.io/llnl/radiuss:${name}-${maj_ver}-ubuntu-22.04"
    spec="%${name}@${ver}"
fi

dockerfile_name="dockerfile_$tag_name"

sed -e "s/<VER>/$ver/g" \
    -e "s/<MAJ_VER>/$maj_ver/g" \
    -e "s/<NAME>/$name/g" \
    -e "s/<SPEC>/$spec/g" \
    -e "s@<IMAGE>@$image@g" dockerfile.in > "$dockerfile_name"
