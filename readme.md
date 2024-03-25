# ML4G
This repo contains the instructions for the ML for Good bootcamp, for students and researchers interested in AI safety.

The program is aimed at beginners in machine learning, but is quite ambitious, and we hope that even advanced students will enjoy participating in this program.

# Curriculum ML4Good

We draw inspiration from the redwood mlab, and ARENA, both of which focuses mainly on the ML engineering part.
However there are a lot more workshops on strategy, goverance and conceptual AI safety during the ML4G.

## For teachers, contributors and assistants

To setup your machine for adding and editing the workshops, follow these steps:
```sh
# Create a virtual environemnt, for instance with venv.
python -m venv .env
# Activate the venv
source .env/bin/activate
# Install all the dependencies. If you don't have a GPU, start by installing pytorch without GPU support
# pip3 install torch  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# Setup the pre-commit hooks
pre-commit install
```

The pre-commit hooks will automatically format and **remove the output** of the notebooks before a commit. It will also prevent commiting large files.
You can also run them manualy at any time on staged files with `pre-commit run`.
If the hooks make any change, it will cancel the commit and add the changes as unstagged changes.

### Adding new workshops

New workshops go in `workshops/`. They should be jupyter notebooks, possibly with auxilary files.
- If they need auxilary files, check how it was done for the [tensors workshop](./workshops/tensors/tensors.ipynb). You will need code similar to that of the first code cell of the notebook.
- Every cell output will be removed from the notebook when you commit (due to pre-commit). This is great for the health of the repo, but if you are still experimentating and developping the notebook you might want to remember it.
- Notebooks are then availaible on google colab with `https://colab.research.google.com/github/EffiSciencesResearch/ML4G/blob/main/<PATH TO NOTEBOOK IN THE GIT>`
- It's helpful if you add this nice badge at the start of the notebook, replacing `<PATH TO NOTEBOOK IN THE GIT>` with `workshops/….ipynb`:
    ```html
    <a href="https://colab.research.google.com/github/EffiSciencesResearch/ML4G/blob/main/<PATH TO NOTEBOOK IN THE GIT>" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    ```
