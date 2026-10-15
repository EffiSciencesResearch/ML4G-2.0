"""Run workshop notebooks top to bottom in Google's Colab runtime image, on Modal.

    modal run -m meta.notebook_tests.modal_app
    modal run -m meta.notebook_tests.modal_app --notebooks workshops/tensors/tensors.ipynb

Writes report.md, failed.txt and the executed notebooks to --out, and exits non-zero if any
notebook failed. Set COLAB_RELEASE to test a specific Colab release instead of the latest.
"""

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import modal

# Older Modal image builders pip-install their own pinned packages (protobuf, aiohttp...)
# into the image, so the test environment would no longer be Colab's.
os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")

COLAB_IMAGE = "us-docker.pkg.dev/colab-images/public/runtime"
SECRET_NAME = "ml4g-notebook-tests"
CELL_TIMEOUT_S = 30 * 60

# The solution notebooks taught at bootcamps. Exercise variants (_normal, _hard)
# are generated from them, so they are not tested separately.
DEFAULT_NOTEBOOKS = [
    "workshops/agents/agents.ipynb",
    "workshops/induction_heads/induction_heads.ipynb",
    "workshops/reasoning_models_faithfullness/reasoning_models_faithfullness.ipynb",
    "workshops/rlhf/rlhf.ipynb",
    "workshops/tensors/tensors.ipynb",
    "workshops/transformer/transformer.ipynb",
    "workshops/transformer_interp/solutions.ipynb",
]


def latest_colab_release() -> str:
    """Name of the release behind the Colab runtime image's `latest` tag."""
    registry, _, repo = COLAB_IMAGE.partition("/")
    with urllib.request.urlopen(f"https://{registry}/v2/{repo}/tags/list", timeout=30) as r:
        manifests = json.load(r)["manifest"].values()
    for tags in (m["tag"] for m in manifests):
        release = next((t for t in tags if t.startswith("release-")), None)
        if "latest" in tags and release:
            return release
    raise RuntimeError(f"No 'latest' tag for {COLAB_IMAGE}: set COLAB_RELEASE to a release tag")


# Resolved on each run, so tests always target the Colab students get today.
# Modal rebuilds the image only when Google publishes a new release.
if modal.is_local():
    COLAB_RELEASE = os.environ.get("COLAB_RELEASE") or latest_colab_release()
else:
    COLAB_RELEASE = os.environ["COLAB_RELEASE"]

image = (
    modal.Image.from_registry(f"{COLAB_IMAGE}:{COLAB_RELEASE}")
    .entrypoint([])
    .env({"COLAB_RELEASE": COLAB_RELEASE})
    .add_local_file(
        Path(__file__).parent / "colab_stubs.py",
        "/root/.ipython/profile_default/startup/00-colab-stubs.py",
    )
    .add_local_python_source("meta")
)
# include_source=False: by default Modal uploads every file of the `meta` package, including
# gitignored credentials, into containers where the agents notebook runs shell commands.
# add_local_python_source above uploads the .py files only.
app = modal.App("ml4g-notebook-tests", include_source=False)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@app.function(
    image=image,
    # Free Colab: 2 vCPUs and about 12.7 GB of RAM. GPU notebooks get a T4 per call.
    # Hard limits (request, limit): by default Modal lets containers burst far above.
    cpu=(2, 2),
    memory=(12_000, 13_000),
    timeout=3 * 60 * 60,
    startup_timeout=20 * 60,
    # pip installs from one notebook must not leak into the next.
    single_use_containers=True,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def run_notebook(source: str) -> dict:
    """Execute a notebook like a student pressing "Run all" on Colab."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    start = time.monotonic()
    nb = nbformat.reads(source, as_version=4)
    current_cell = None

    def on_cell_start(cell_index, **_):
        nonlocal current_cell
        current_cell = cell_index

    # Colab opens notebooks in /content, and some notebooks rely on it.
    Path("/content").mkdir(exist_ok=True)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT_S,
        # The Colab kernel imports a lot, and the image is lazily loaded.
        startup_timeout=5 * 60,
        kernel_name="python3",
        resources={"metadata": {"path": "/content"}},
        on_cell_start=on_cell_start,
    )
    result = {"passed": True}
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError, DeadKernelError) as e:
        error = (
            f"{e.ename}: {e.evalue}"
            if isinstance(e, CellExecutionError)
            else f"{type(e).__name__}: {e}"
        )
        if current_cell is None:
            result = {"passed": False, "error": f"Kernel failed to start: {error}"}
        else:
            cell = nb.cells[current_cell]
            errors = [o for o in cell.get("outputs", []) if o.output_type == "error"]
            result = {
                "passed": False,
                "failed_cell": current_cell,
                "failed_source": cell.source,
                "error": error,
                "traceback": strip_ansi("\n".join(errors[-1].traceback)) if errors else "",
            }
    result["minutes"] = (time.monotonic() - start) / 60
    result["executed"] = nbformat.writes(nb)
    return result


def needs_gpu(source: str) -> bool:
    # The field Colab reads to pick the runtime type.
    return json.loads(source).get("metadata", {}).get("accelerator") == "GPU"


def code_block(text: str, language: str = "") -> list[str]:
    # A fence longer than any backtick run inside, which notebooks do contain.
    longest = max((len(run) for run in re.findall("`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + language, text, fence]


def report(results: dict[str, dict]) -> str:
    n_failed = sum(not r["passed"] for r in results.values())
    lines = [
        "# Notebook tests on Colab",
        "",
        f"Colab runtime `{COLAB_RELEASE}`: "
        + (f"**{n_failed} of {len(results)} notebooks failed.**" if n_failed else "all passed."),
        "",
        "| | Notebook | Minutes | Error |",
        "|---|---|---|---|",
    ]
    for path, r in results.items():
        minutes = f"{r['minutes']:.1f}" if "minutes" in r else ""
        error = r.get("error", "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {'✅' if r['passed'] else '❌'} | `{path}` | {minutes} | {error} |")
    for path, r in results.items():
        if r["passed"]:
            continue
        lines += ["", f"<details><summary>❌ <code>{path}</code></summary>", ""]
        if "failed_cell" in r:
            lines += [f"In cell {r['failed_cell'] + 1} of the notebook:", ""]
            lines += [*code_block(r["failed_source"], "python"), ""]
        lines += [*code_block(r.get("traceback") or r["error"]), "", "</details>"]
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(notebooks: str = "", out: str = "notebook-test-results"):
    paths = notebooks.split(",") if notebooks else DEFAULT_NOTEBOOKS
    print(f"Testing {len(paths)} notebooks on Colab runtime {COLAB_RELEASE}", flush=True)

    # All notebooks run in parallel, each in its own container.
    run_notebook_gpu = run_notebook.with_options(gpu="T4")
    calls = {}
    for path in paths:
        source = Path(path).read_text()
        calls[path] = (run_notebook_gpu if needs_gpu(source) else run_notebook).spawn(source)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for path, call in calls.items():
        try:
            results[path] = call.get()
        except Exception as e:
            # The container itself failed: timeout, out of memory...
            results[path] = {"passed": False, "error": f"{type(e).__name__}: {e}"}
        if executed := results[path].pop("executed", None):
            (out_dir / path.replace("/", "__")).write_text(executed)
        print(f"{'✅' if results[path]['passed'] else '❌'} {path}", flush=True)

    summary = report(results)
    (out_dir / "report.md").write_text(summary)
    failed = [path for path, r in results.items() if not r["passed"]]
    (out_dir / "failed.txt").write_text("".join(f"{path}\n" for path in failed))
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as f:
            f.write(summary)
    print(summary)
    if failed:
        raise SystemExit(1)
