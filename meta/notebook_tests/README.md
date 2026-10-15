# Notebook Tests

Runs the workshop notebooks top to bottom in Google's [Colab runtime image](https://us-docker.pkg.dev/colab-images/public/runtime), on [Modal](https://modal.com), and reports which cells break. It catches what breaks notebooks without anyone touching them: Colab upgrading its preinstalled packages, unpinned dependencies moving on, models and datasets disappearing.

It runs every Monday from GitHub Actions ([`notebook-tests.yml`](../../.github/workflows/notebook-tests.yml)), on `master` only. To run it now: Actions → *Notebook tests on Colab* → *Run workflow*. Each run shows a report on its summary page: which notebooks failed, the failing cell and its traceback. When a notebook fails, a Slack message links to it.

GitHub disables scheduled workflows of public repositories after 60 days without commits. If the Monday runs stop, re-enable the workflow from the Actions tab.

## How it works

- `modal_app.py` — looks up the latest Colab runtime release, then runs each notebook in a fresh container built from it, all in parallel. Containers match free Colab: 2 vCPUs and about 12.7 GB of RAM, plus a T4 GPU for notebooks that ask for one. Each cell gets 30 minutes.
- `colab_stubs.py` — IPython startup file in the test kernel, replacing what needs a human on Colab:
  - `userdata.get` reads the Modal secret instead of the Colab secrets sidebar.
  - `input()` returns `""`, as if the human pressed Enter. In the agents notebook, this approves every command the agent proposes, inside the throwaway container. The container holds the OpenRouter key, so give it a dedicated key with a low credit limit.

Notebooks run from `/content` with `google.colab` importable, so `IN_COLAB` is true and their own `%pip install` and download cells run as they do for students.

The tested notebooks are `DEFAULT_NOTEBOOKS` in `modal_app.py`: the solution notebooks taught at bootcamps. Exercise variants are generated from them, so they are not tested separately.

## GPU notebooks

A notebook runs on a GPU when its metadata has `"accelerator": "GPU"`, the field Colab reads to open it on a GPU runtime. `notebook_tools clean` keeps this field, so setting it once serves both students and tests. Set it in Colab (Runtime → Change runtime type → T4 GPU, then save to GitHub), or add it to the notebook's top-level `metadata` by hand.

## Secrets

| Secret | Stored in | Used for | Owner / destination | To rotate |
|---|---|---|---|---|
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | GitHub environment `notebook-tests` | running the notebooks on Modal | Modal workspace `ddorn` (Diego) | Modal dashboard → Settings → API Tokens → new token, then `gh secret set <NAME> --env notebook-tests` |
| `SLACK_WEBHOOK_URL` | GitHub environment `notebook-tests` | failure messages | Slack app *ML4G notebook tests*, posting to Diego's DMs | Slack app → Incoming Webhooks → new webhook, then `gh secret set SLACK_WEBHOOK_URL --env notebook-tests` |
| `OPENROUTER_API_KEY` | Modal secret `ml4g-notebook-tests` | LLM calls in the notebooks | Diego's OpenRouter account, a dedicated key with a credit limit | OpenRouter → Keys → new key, then `modal secret create --force ml4g-notebook-tests OPENROUTER_API_KEY=...` |

The `notebook-tests` environment only allows `master` (Settings → Environments → *Deployment branches and tags*), so jobs from any other branch cannot read its secrets, even if they edit the workflow. Keep these as environment secrets: repository secrets are readable from every branch.

To set this up in a new repository or Modal workspace: create the environment with that branch rule, then create each secret as in the *To rotate* column.

## Run from your machine

With your own Modal token (`modal setup`):

```shell
# The default notebooks, from your working copy
uv run --only-group notebook-tests modal run -m meta.notebook_tests.modal_app

# Some notebooks
uv run --only-group notebook-tests modal run -m meta.notebook_tests.modal_app --notebooks workshops/tensors/tensors.ipynb,workshops/rlhf/rlhf.ipynb
```

The report and the executed notebooks, with their outputs, land in `notebook-test-results/`. Set `COLAB_RELEASE` to test a specific Colab release instead of the latest.

## Troubleshooting

- The first run after a new Colab release spends about 7 minutes building the 23 GB image. Later runs reuse it.
- A cell that waits on the Colab frontend (e.g. `google.colab.output.eval_js`) hangs until the cell timeout. Add a stub for it in `colab_stubs.py`.
- Notebooks that download auxiliary files from GitHub get them from `master`.
