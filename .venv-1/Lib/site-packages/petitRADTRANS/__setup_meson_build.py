"""Script to automatically make meson builds in submodules.
Since meson does not support wildcards (https://mesonbuild.com/FAQ.html#why-cant-i-specify-target-files-with-a-wildcard)
this is a way of quickly setup petitRADTRANS to be sure that every python files are included in the build.
"""
import copy
import os


def _build_python_files_hierarchy(parent):
    python_files = []
    children = os.listdir(parent)

    # Search for Python files in the parent directory
    for child in children:
        child = os.path.abspath(os.path.join(parent, child))

        # Search as well inside the parent directories
        if os.path.isdir(child):
            python_files.append(_build_python_files_hierarchy(child))

        # Add Python file
        if child.endswith(".py"):
            python_files.append(child)

    # Only return something if at least one Python file was found in the parent or in one of its subdirectories
    if len(python_files) > 0:
        return python_files

    return None


def _get_directories_dict(files, paths=None):
    if paths is None:
        paths = {}

    subdirectories = set()
    current_path = None
    has_subdirectories = False

    for file in files:
        if file is None:
            continue
        elif isinstance(file, list):
            paths, sub_dir = _get_directories_dict(file, paths)
            subdirectories = subdirectories.union(sub_dir)

            if len(subdirectories) > 0:
                has_subdirectories = True
        else:
            current_path, f = file.rsplit(os.path.sep, 1)
            subdirectories.add(current_path)

            if current_path not in paths:
                paths[current_path] = {'files': [f]}
            else:
                paths[current_path]['files'].append(f)

    if current_path is None:
        raise ValueError(f"No path found when looking for paths in {files}")

    if has_subdirectories:
        sub_dir = copy.deepcopy(subdirectories)
        sub_dir.discard(current_path)

        paths[current_path]['subdirectories'] = sub_dir
    else:
        paths[current_path]['subdirectories'] = None

    return paths, subdirectories


def init_meson_build():
    root = os.path.abspath(os.path.dirname(__file__)).rsplit(os.path.sep, 1)[1]

    python_files = _build_python_files_hierarchy(os.path.abspath(os.path.dirname(__file__)))
    directories_dict, _ = _get_directories_dict(python_files)

    for directory, content in directories_dict.items():
        # Get meson.build file lines
        subdir_str = ''
        install_str = ''

        # List the subdirectories to search for other meson.build
        if content['subdirectories'] is not None:
            subdir_str = ''

            for subdirectory in content['subdirectories']:
                subdir_str += f"subdir('{subdirectory.rsplit(os.path.sep, 1)[1]}')\n"

            subdir_str += '\n'

        # Python source files to install
        if content['files'] is not None:
            install_str = "py.install_sources([\n"

            for file in content['files']:
                install_str += f"    '{file}',\n"

            install_str += "    ],\n"

            if directory[len(directory) - len(root):] != root:  # if directory is not root, add the 'subdir' keyword
                subdir = '/'.join((root, directory.rsplit(root + os.path.sep, 1)[1]))
                install_str += f"    subdir: '{subdir}'\n"
            else:
                install_str += f"    subdir: '{root}'\n"

            install_str += ")\n"

        lines = subdir_str + install_str

        meson_build_file = os.path.join(directory, 'meson.build')

        if os.path.isfile(meson_build_file):
            print(f"Updating meson build file '{meson_build_file}'...", end="")
        else:
            print(f"Writing new meson build file '{meson_build_file}'...", end="")

        with open(meson_build_file, 'w') as f:
            f.write(lines)

        print(" Done")

    print("petitRADTRANS ready for meson install")
