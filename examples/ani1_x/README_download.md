# Recovering from an empty Figshare download

Older ANI-1x and Transition1x download scripts could report success while
saving a zero-byte file after a Figshare HTTP 202 challenge. The corrected
scripts use Figshare's public API, and the downloader rejects HTTP responses
other than 200 or 206.

Existing completed files are still reused without a checksum. Before retrying,
check `ani1x-release.h5` or `transition1x-release.h5` in your output directory.
Remove only a confirmed invalid zero-byte destination file, then rerun the
corresponding download script. Do not delete valid datasets or interrupted
`.part` files: partial downloads can be resumed.
