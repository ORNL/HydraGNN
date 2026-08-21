# Dataset download utilities

`hydragnn.utils.datasets.download` provides reusable dataset downloads without
requiring `wget` or shell command construction. Downloads are written to a
`.part` file, resumed with HTTP range requests, and atomically renamed after
completion. An optional SHA-256 digest verifies both cached and new files.

The same module safely extracts tar archives by rejecting path traversal,
links, and special-file entries. Tests use mocked responses and local archives;
HydraGNN CI does not depend on external dataset servers.

It can also be used from the command line:

```bash
python -m hydragnn.utils.datasets.download URL DESTINATION \
    [--sha256 DIGEST] [--extract-to DIRECTORY] [--remove-archive]
```

The shared transport is used by MPTrj, OMat24, ANI-1x, Transition1x, OC20,
OC22, ODAC23, OMol25, OP26, and QM7-X. Their example entry points retain
dataset-specific URLs, filenames, selection, and post-download organization.
Site-specific proxy configuration is left to the user's environment rather
than embedded in repository scripts. Specialized discovery-based downloaders
such as Alexandria and Nabla2 retain their purpose-built workflows.

On OLCF systems whose compute nodes require the CCS proxy, source the optional
facility helper before invoking any dataset downloader:

```bash
source scripts/hpc/olcf/proxy-env.sh
```

Normal pull-request tests run a complete download, resume, redirect, checksum,
CLI, and extraction flow against a local HTTP server. They also verify that
every migrated example calls the shared transport. No external service is
required. A separate weekly/manual workflow probes only the first 64 KiB from
each live dataset endpoint; run it locally with
`HYDRAGNN_RUN_NETWORK_TESTS=1 pytest -m network`.
