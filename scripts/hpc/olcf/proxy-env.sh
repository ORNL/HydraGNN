#!/bin/bash
# Copyright (c) 2026, Oak Ridge National Laboratory
# All rights reserved.
#
# This file is part of HydraGNN and is distributed under a BSD 3-clause
# license. For the licensing terms see the LICENSE file in the top-level
# directory.
#
# SPDX-License-Identifier: BSD-3-Clause

# Source this optional helper before downloading datasets from an OLCF system
# whose compute nodes require the CCS proxy:
#
#   source scripts/hpc/olcf/proxy-env.sh

export all_proxy="socks://proxy.ccs.ornl.gov:3128/"
export ftp_proxy="ftp://proxy.ccs.ornl.gov:3128/"
export http_proxy="http://proxy.ccs.ornl.gov:3128/"
export https_proxy="http://proxy.ccs.ornl.gov:3128/"
export no_proxy="localhost,127.0.0.0/8,*.ccs.ornl.gov"
