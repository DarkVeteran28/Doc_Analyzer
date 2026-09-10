"""
Shared pytest configuration.

All test artifact directories (ChromaDB, BM25 indexes) are written under
TEST_ARTIFACTS so generated data stays in one place instead of scattering
named folders across the project root.
"""
import os

TEST_ARTIFACTS = "test_artifacts"
