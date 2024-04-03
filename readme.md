# ML4G
This repo contains the instructions for the ML for Good bootcamp, for students and researchers interested in AI safety.

The program is aimed at beginners in machine learning, but is quite ambitious, and we hope that even advanced students will enjoy participating in this program.

## Curriculum ML4Good

We draw inspiration from the redwood mlab, and ARENA, both of which focuses mainly on the ML engineering part.
However there are a lot more workshops on strategy, goverance and conceptual AI safety during the ML4G.

## For teachers, contributors and assistants

To setup your machine for adding and editing the workshops, follow these steps:

#### Linux setup
```sh
# Create a virtual environemnt (in the current folder), for instance with venv.
python -m venv .env

# Activate the venv
source .env/bin/activate

# Install all the dependencies. If you don't have a GPU, start by installing pytorch without GPU support
# pip3 install torch  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Setup the pre-commit hooks
pre-commit install
```

#### Windows setup
(equivalent according to GPT4)
```cmd
REM Create a virtual environment (in the current folder), for instance with venv.
python -m venv .env

REM Activate the venv
.env\Scripts\activate

REM Install all the dependencies. If you don't have a GPU, start by installing pytorch without GPU support
REM pip install torch  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

REM Setup the pre-commit hooks
pre-commit install
```

### Pre-commit hooks

> ⚠ It is very important that you install the pre-commit hook!

The [pre-commit](https://pre-commit.com/) hooks are run every time you commit. They help maintain the quality of the repository. They:
- Remove the output of the notebooks before commiting
    - Note: that this great for the health of the repo, but if you are still experimentating and developping the notebook you might want to keep the output of your cells. If so, you need to not commit until you are ready to remove the output.
- Format the notebooks with black
- Prevent you from commiting files larger than 500kb
- Add and sync the 'add to colab' badge at the start of the notebook

You can also run them manualy at any time on staged files with `pre-commit run` or on all files with `pre-commit run --all-files`.
If the hooks make any change, it will cancel the commit and add the changes as unstagged changes.
You can then review them, `git add` the changes and commit again.


### Adding new workshops

#### Guidelines
1. Create a folder for the workshop with using snake case such as `workshops/logit_lens`.
1. Create a notebook for the solution, with the same name as the folder, such as `workshops/logit_lens/logit_lens.ipynb`.
1. Write the learning objectives at the top of the notebook.
1. At the end of the notebook, add a recap of what was learned.
1. If you need auxilary files, check how it was done for the [tensors workshop](./workshops/tensors/tensors.ipynb). You will need code similar to that of the first code cell of the notebook, replacing every instance of `tensor` by the name of your workshop.
    - If you need to add files more than 500kb, discuss with the team first to find a solution.

#### Creating exercises notebooks
We create notebooks of different difficulty levels, automatically from the solutions notebook. In the code cells of the solution notebook, you can add `# Hide: <name>` comments such to hide all the following lines until the next `# Hide:` comment.

To generate the notebooks, run the following command:
```sh
python meta/tools.py sync <path_to_notebook>.ipynb
```

Here name `<name>` can be any sequence of letters (lowercase), but
we usually use `hard`, `normal`, `all`, `none` or `solution`.


- `# Hide: all` will hide the following lines for every level of exercise notebook.
- `# Hide: none` cancels the effect of any previous `# Hide` primitive.
- `# Hide: hard` will hide the following lines only in the notebook `<workshop_name>_hard.ipynb`.
- `# Hide: normal` will hide the following lines only in the notebook `<workshop_name>_normal.ipynb`.
- `# Hide: solution` will hide the following lines only in the collapsible solution cell that is generated automatically after the code cell, with the full solution.

Note:
1. You can use any name in `# Hide: <name>`, and it will create a new notebook with that name.
1. You can use multiple names in a hide comment, such as `# Hide: hard, solution`, which will then be shown only in the `normal` notebook, and note in code cells.
1. A `# Hide:` directives hides every line until the end of the cell, or until the next `# Hide:` directive, which always resets what is hidden.
1. The `# Hide: <name>` comments must be on their own line.
1. The hide directive is replace by `...` if none were already present just before.

Example:
```python
print("This will be shown in all notebooks")
# Hide: hard
for i in range(10):
    print("This will be shown only in the normal notebook")
    # Hide: all
    print("This will be shown only in the solution")
# Hide: none
print("Success 🎉")
```
This would create two other notebooks. A hard notebook with
```python
print("This will be shown in all notebooks")
...
print("Success 🎉")
```
and a normal notebook with
```python
print("This will be shown in all notebooks")
for i in range(10):
    print("This will be shown only in the normal notebook")
    ...
print("Success 🎉")
```
And both will have a collapsible solution cell bellow as such
<details>
<summary>Show solution</summary>

```python
print("This will be shown in all notebooks")
for i in range(10):
    print("This will be shown only in the normal notebook")
    print("This will be shown only in the solution")
print("Success 🎉")
```
</details>
