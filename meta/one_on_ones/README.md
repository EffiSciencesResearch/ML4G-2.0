# One-on-Ones

NetworkX-based pairing algorithm used by the **One On One Scheduler** page of the [internal web app](../web/README.md). Lives here as a standalone module because the algorithm has nothing to do with streamlit; the page just imports `make_pairing_graph` from `pairing.py`.

No CLI — the web UI is the only interface.
