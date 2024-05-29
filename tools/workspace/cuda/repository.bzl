load(
    "@drake//tools/workspace:execute.bzl",
    "execute_and_return",
)
load(":manifest-22.04.bzl", MANIFEST_2204 = "MANIFEST")

# The list of debs to unpack and make available via our repository rule.
_PACKAGES_2204 = [
    "cuda-cccl",
    "cuda-crt",
    "cuda-cudart",
    "cuda-cudart-dev",
    "cuda-nvcc",
    "cuda-nvvm",
    "cuda-sanitizer",
    "libnpp",
    "libnpp-dev",
    "libnvjpeg",
    "libnvjpeg-dev",
    "libcusolver",
    "libcusolver-dev",
    "libcublas",
    "libcublas-dev",
    "libcusparse",
    "libcusparse-dev",
]

UBUNTU_RELEASE = "22.04"

def _impl(repo_ctx):
    name = repo_ctx.attr.name
    filenames = repo_ctx.attr.filenames
    mirrors = repo_ctx.attr.mirrors
    sha256s = repo_ctx.attr.sha256s
    build_file = repo_ctx.attr.build_file
    manifest_file = repo_ctx.attr.manifest_file

    # Download all of the debs.  We should do all downloads prior to any other
    # exectute steps so that rule restarts on cache misses are efficient.
    for filename, sha256 in zip(filenames, sha256s):
        if not sha256:
            # We do not permit an empty checksum; empty means "don't care".
            sha256 = "0" * 64
        repo_ctx.download(
            url = [mirror.format(basename = filename) for mirror in mirrors],
            output = filename,
            sha256 = sha256,
        )

    # Extract all of the debs.
    multi_sh = repo_ctx.path(Label("@drake//tools/workspace/cuda:multi_sh.py"))
    command = ["/usr/bin/python3", multi_sh]
    for filename in filenames:
        command += ["/usr/bin/dpkg-deb", "-x", filename, ".", "&&"]
    result = execute_and_return(repo_ctx, command)
    if result.error:
        fail(result.error)

    # Remove useless files.
    command = ["sh", "-c", "rm *.deb usr/local/*/*/*.a"]
    result = execute_and_return(repo_ctx, command)
    if result.error:
        fail(result.error)

    # Add in the build files.
    repo_ctx.symlink(build_file, "BUILD.bazel")
    repo_ctx.symlink(manifest_file, "manifest.bzl")

_cuda_repository_rule = repository_rule(
    attrs = {
        "mirrors": attr.string_list(
            mandatory = True,
            allow_empty = False,
        ),
        "filenames": attr.string_list(
            mandatory = True,
            allow_empty = False,
        ),
        "sha256s": attr.string_list(
            mandatory = True,
            allow_empty = False,
        ),
        "build_file": attr.label(
            mandatory = True,
            allow_files = True,
        ),
        "manifest_file": attr.label(
            mandatory = True,
            allow_files = True,
        ),
    },
    implementation = _impl,
)

def cuda_repository(
        name,
        mirrors = None,
        **kwargs):
    if UBUNTU_RELEASE == "22.04":
        MANIFEST = MANIFEST_2204
        PACKAGES = _PACKAGES_2204
    else:
        fail("Unsupported Ubuntu")
    cuda_mirrors = mirrors["cuda_debs"]
    filenames = [MANIFEST[x]["Filename"] for x in PACKAGES]
    sha256s = [MANIFEST[x]["SHA256"] for x in PACKAGES]
    build_file = "//tools/workspace/cuda:package.BUILD.bazel"
    manifest_file = "//tools/workspace/cuda:manifest-{}.bzl".format(
        UBUNTU_RELEASE,
    )
    _cuda_repository_rule(
        name = name,
        mirrors = cuda_mirrors,
        filenames = filenames,
        sha256s = sha256s,
        build_file = build_file,
        manifest_file = manifest_file,
    )
