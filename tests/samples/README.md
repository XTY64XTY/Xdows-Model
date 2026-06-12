# Safe Model Test Samples

This directory is intentionally kept free of malware samples.

Recommended local samples:

- A freshly built `Xdows-Model-Caller.exe` for PE parsing and safe model inference.
- A signed Windows binary for trusted benign input checks.
- A small unsigned local helper executable for unknown-publisher behavior.
- An EICAR text file only when explicitly testing antivirus detection paths outside the driver blocking flow.

Do not commit live malware, weaponized scripts, credentials, or customer files.
