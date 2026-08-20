##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
from urllib.request import Request, urlopen

import pytest

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.getenv("HYDRAGNN_RUN_NETWORK_TESTS") != "1",
        reason="set HYDRAGNN_RUN_NETWORK_TESTS=1 to probe live dataset URLs",
    ),
]


ENDPOINTS = {
    "mptrj": "https://ndownloader.figshare.com/files/41619375",
    "transition1x": "https://figshare.com/ndownloader/files/36035789",
    "ani1x": "https://springernature.figshare.com/ndownloader/files/18112775",
    "oc20": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
    "oc22": "https://materials.colabfit.org/dataset-original/DS_jgaid7espcoc_0",
    "odac23": "https://dl.fbaipublicfiles.com/dac/datasets/extxyz_val.tar.gz",
    "omat24": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz",
    "omol25": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz",
    "op26": "https://dl.fbaipublicfiles.com/opencatalystproject/data/op26/op26_train_val_260108.tar.gz",
    "qm7x": "https://zenodo.org/api/records/3905361/files-archive",
}


@pytest.mark.parametrize(("dataset", "url"), ENDPOINTS.items())
def pytest_live_dataset_endpoint_supports_bounded_probe(dataset, url):
    request = Request(
        url,
        headers={"Range": "bytes=0-65535", "User-Agent": "HydraGNN-CI/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        content = response.read(65536)
        final_url = response.geturl()

    assert content, f"{dataset} returned no data"
    assert final_url.startswith("https://"), final_url
