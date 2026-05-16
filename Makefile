# Set uv to either /root/.local/bin/uv or uv depending on which one exists
# That's just because I don't have uv in the path on my deployment server
UV := uv

run:
	PYTHONPATH=. $(UV) run streamlit run meta/web/main.py --server.port 8991

run-changelog:
	$(UV) run python -m meta.drive_changelog

.PHONY: run run-changelog
