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

MPTrj and OMat24 are the first consumers. Site-specific proxy configuration is
left to the user's environment rather than embedded in repository scripts.
