"""Builders that turn each benchmark's official release into Harbor task directories.

No benchmark content is shipped with this repository: every builder reads the official sources (a git clone, a Hugging
Face download, or a Harbor registry dataset) and writes the task directories under ``datasets/``. See README.md.
"""
