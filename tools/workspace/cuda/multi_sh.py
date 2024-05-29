"""Runs multiple subproesses in parallel, failing if any one fails.

Invoke this program like
/usr/bin/python3 multi_sh.py \
  tar xvfz foo.tgz && \
  tar xvfz bar.tgz && \
  tar xvfz bar.tgz &&

Note that even the final command must be followed by an '&&'.

This program is intended for use by repository rules, not directly by users.
"""

# N.B. We use "&&" as the command separator as a token that generally would not
# otherwise appear within any given command line.

from multiprocessing.pool import Pool
import subprocess
import sys


def main():
    argv = list(sys.argv[1:])
    commands = []
    while argv:
        index = argv.index("&&")
        assert index >= 0
        one_command = argv[:index]
        argv = argv[index + 1:]
        commands.append(one_command)
    try:
        pool = Pool()
        mapper = pool.map
    except PermissionError as e:
        # Sometimes our runtime environment doesn't support multiprocessing,
        # e.g., in chroot jails.  Fall back to serial operation in that case.
        mapper = map
    results = mapper(subprocess.run, commands)
    for x in results:
        x.check_returncode()
    return 0


if __name__ == "__main__":
    sys.exit(main())
