# Set uv to either /root/.local/bin/uv or uv depending on which one exists
# That's just because I don't have uv in the path on my deployment server
UV := uv

run:
	$(UV) run streamlit run meta/web.py --server.port 8991

run-changelog:
	$(UV) run python meta/drive_changelog.py

.PHONY: run run-changelog
