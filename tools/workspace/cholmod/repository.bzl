# -*- python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def cholmod_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "DrTimothyAldenDavis/SuiteSparse",
        commit = "v5.13.0",
        sha256 = "59c6ca2959623f0c69226cf9afb9a018d12a37fab3a8869db5f6d7f83b6b147d",  # noqa
        build_file = ":package.BUILD.bazel",
        mirrors = mirrors,
        patches = [
            ":patches/config.patch",
        ],
    )
