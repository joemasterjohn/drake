#!/usr/bin/env python3

"""Regenerates the manifest.bzl list of packages and checksums."""

import gzip
import sys
import urllib.request

# This is NVidia's package listing for their download site.
# When upgrading Ubuntu versions and supporting both versions as an
# intermediate step, please use the URL which references the later version of
# Ubuntu.
URL = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz"  # noqa

CUDA_VERSION = "12-5"  # Use `-` not `.` for filename matching.
# This is the CUDA packages and their versions that we want to use.
# From
#    https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-major-component-versions
# or for less-that-latest versions, browse through here instead:
#    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
# Minimum NVIDIA Linux driver for this version: 550.54.15
PACKAGE_VERSION_MAP = {
    "cuda-cccl": "12.5.39",
    "cuda-crt": "12.5.82",
    "cuda-cudart": "12.5.82",
    "cuda-cudart-dev": "12.5.82",
    "cuda-nvcc": "12.5.82",
    "cuda-nvvm": "12.5.82",
    "cuda-sanitizer": "12.5.81",
    "libnpp": "12.3.0.159",
    "libnpp-dev": "12.3.0.159",
    "libnvjpeg": "12.3.2.81",
    "libnvjpeg-dev": "12.3.2.81",
    "libcusolver": "11.6.3.83",
    "libcusolver-dev": "11.6.3.83",
    "libcublas": "12.5.3.2",
    "libcublas-dev": "12.5.3.2",
    "libcusparse": "12.5.1.3",
    "libcusparse-dev": "12.5.1.3",
}


def expected_filename(package_name):
    """Returns the filename of a package based on its wanted version."""
    package_version = PACKAGE_VERSION_MAP[package_name]
    return f"{package_name}-{CUDA_VERSION}_{package_version}-1_amd64.deb"


def extract_manifest(packages_data):
    """Given unzipped Packages.gz contents, returns the MANIFEST dict to be
    written into manifest.bzl."""
    wanted_filenames = {
        expected_filename(package_name): package_name
        for package_name in PACKAGE_VERSION_MAP
    }
    result = {}
    for one_package in packages_data.split("\n\n"):
        lines = one_package.split("\n")
        if not one_package:
            continue
        # Extract "Filename: " and "SHA256: " for each entry.
        package_details = {}
        for key in ["Filename", "SHA256"]:
            prefix = key + ": "
            if key == "Filename":
                # Prune the leading "./" from filenames.
                prefix += "./"
            match = [x for x in lines if x.startswith(prefix)]
            assert len(match) == 1, one_package
            package_details[key] = match[0][len(prefix):]
        # Skip entries that to not match our desired version.
        filename = package_details["Filename"]
        if filename in wanted_filenames:
            package_name = wanted_filenames[filename]
            result[package_name] = package_details

    assert len(result) == len(
        PACKAGE_VERSION_MAP
    ), f"Missing packages: {set(PACKAGE_VERSION_MAP) - set(result)}"

    return result


def make_bzl(manifest):
    """Given the MANIFEST dict, return the manifest.bzl file contents."""
    result = [
        "# GENERATED FILE " + "DO NOT EDIT",
        "# To update this, run ./tools/workspace/cuda/manifest_regenerate.py.",
        "",
        "CUDA_VERSION = \"{}\"".format(CUDA_VERSION.replace("-", ".")),
        "",
        "MANIFEST = {",
    ]
    for name in sorted(manifest.keys()):
        result.extend(
            [
                '    "{}": {{'.format(name),
                '        "Filename": "{}",'.format(manifest[name]["Filename"]),
                '        "SHA256": "{}",  # {}'.format(
                    manifest[name]["SHA256"], "noqa"
                ),
                '    },',
            ]
        )
    result.append("}")
    return "\n".join(result) + "\n"


def main():
    # Read NVidia's Packages.gz list of debs + checksums.
    gzipped_contents = urllib.request.urlopen(URL).read()
    listing = gzip.decompress(gzipped_contents).decode("utf-8")

    # Scrape the data that we want.
    manifest = extract_manifest(listing)

    # Format it into buildifer-formatted bzl file.
    bzl = make_bzl(manifest)

    # Overwrite Anzu's source manifest.
    with open("/home/jsonzhou/Desktop/drake/tools/workspace/cuda/manifest-22.04.bzl", "w") as f:
        f.write(bzl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
