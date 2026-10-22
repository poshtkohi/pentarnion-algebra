# Data guides

- [Part I data and coordinate conventions](../part-I/docs/data-guide.md)
- [Part II data and coordinate conventions](../part-II/docs/data-guide.md)

Reference records are stored in each paper's `data/` directory. Numerical runs
write to separate output directories. Compressed CSV trajectories are read
directly without extraction.

[Reference-file checksums](reference_integrity.json) are used by the integrity
tests. See the [numerical notes](numerical-notes.md) when comparing regenerated
results with the paper reference records.
