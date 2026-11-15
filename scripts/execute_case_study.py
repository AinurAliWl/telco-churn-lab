"""Execute the walkthrough in IPython without starting a networked notebook kernel."""

import argparse
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook", nargs="?", type=Path, default=Path("notebooks/01_case_study.ipynb")
    )
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    shell = InteractiveShell.instance()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(
                f"Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}"
            )
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(
                nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata)
            )
        cell.outputs = outputs
        cell.execution_count = count
    notebook.metadata.kernelspec = {
        "display_name": "Python 3.12",
        "language": "python",
        "name": "python3",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, args.notebook)
    print(f"Executed {count} notebook cells successfully: {args.notebook}")


if __name__ == "__main__":
    main()
