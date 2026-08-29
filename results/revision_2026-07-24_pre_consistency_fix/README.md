# Historical revision summaries — pre consistency fix

These files summarize the archived paper run tree executed on July 24, 2026.

- They were produced by reading the archived run outputs with the table collector.
- No simulation was rerun during cleanup.
- The run cards report Python 3.10.12, NumPy 2.2.6, PyTorch 2.5.1+cu121, CUDA, and `git_commit: null`.
- The source snapshot used for those runs predates the covariance/loading consistency correction now integrated into `src/mf_revision/models/market.py`.

Treat the CSVs as revision-history evidence. Corrected results should be generated with `make ridge-rerun` and stored under a new versioned directory.
