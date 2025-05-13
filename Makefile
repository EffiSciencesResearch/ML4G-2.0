# Set uv to either /root/.local/bin/uv or uv depending on which one exists
# That's just because I don't have uv in the path on my deployment server
UV := $(shell command -v uv >/dev/null 2>&1 && echo "uv" || echo "/root/.local/bin/uv")

run:
	$(UV) run streamlit run meta/web.py --server.port 8991

deploy:
	@# Abort if variable is not set, and print error message
	@test -n "$(DEPLOY_HOST)" || (echo "DEPLOY_HOST is not set" && exit 1)
	@test -n "$(DEPLOY_PATH)" || (echo "DEPLOY_PATH is not set" && exit 1)

	@echo "Deploying to $(DEPLOY_HOST):$(DEPLOY_PATH)"
	git ls-files | rsync -azP --files-from=- . $(DEPLOY_HOST):$(DEPLOY_PATH)
	rsync -azP ml4g-web.service $(DEPLOY_HOST):/etc/systemd/system/ml4g-web.service
	rsync -azP .env-prod $(DEPLOY_HOST):$(DEPLOY_PATH)/.env
	ssh $(DEPLOY_HOST) "systemctl daemon-reload && systemctl restart ml4g-web && journalctl -u ml4g-web -f"

.PHONY: deploy run
