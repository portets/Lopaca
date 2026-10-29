# Copy-on-write

Copy-on-write shares immutable pages until mutation requires a private allocation. The private allocation granularity is 96 KiB.
