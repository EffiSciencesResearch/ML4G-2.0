# ML4G
This repo contains the instructions for the ML for Good bootcamp, for students and researchers interested in AI safety.

The program is aimed at beginners in machine learning, but is quite ambitious, and we hope that even advanced students will enjoy participating in this program.

# Curriculum ML4Good

We draw inspiration from the redwood mlab, which focuses mainly on the ML engineering part. In comparison, we will spend more time on conceptual aspects.

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
