# Onramp. Targets are ordered roughly in the sequence you would run them.
# Uses only BSD-compatible tooling so it works with the make and shell that
# ship with macOS (bash 3.2, no GNU coreutils).

PY       := ./.venv/bin/python
PIP      := ./.venv/bin/pip
PYTEST   := ./.venv/bin/pytest
RUFF     := ./.venv/bin/ruff
REPO     ?= /tmp/adobedocs
PROFILE  ?= admin
REGION   ?= us-east-1

.PHONY: help venv load test lint fmt eval-set build deploy ingest local-ingest \
        ask local-ask bench eval citations destroy clean

help:
	@echo "venv         create .venv with python3.12 and install dev deps"
	@echo "load         clone the Adobe docs repo to \$$REPO and sync markdown to \$$BUCKET"
	@echo "eval-set     validate golden-draft.json and write evals/golden.json"
	@echo "test         run pytest with the Null provider"
	@echo "lint         ruff check + format --check"
	@echo "fmt          ruff format"
	@echo "build        build both Lambda container images locally (stage 3)"
	@echo "deploy       cdk deploy (stage 2 onward)"
	@echo "ingest       trigger the deployed event-driven ingest (stage 3)"
	@echo "local-ingest run the identical chunker/embedder on this laptop (stage 2)"
	@echo "ask          query the deployed API (stage 4)"
	@echo "local-ask    query the same logic locally (stage 2)"
	@echo "bench        p50/p95 across four conditions (stage 5)"
	@echo "eval         recall@1, recall@3, refusal rate, latency (stage 2 onward)"
	@echo "citations    HEAD-check a sample of generated Experience League URLs"
	@echo "destroy      cdk destroy and empty the corpus bucket"

venv:
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

load:
	@test -n "$$BUCKET" || { echo "BUCKET is not set. Run: export BUCKET=<your bucket>"; exit 1; }
	rm -rf $(REPO)
	git clone --depth 1 https://github.com/AdobeDocs/experience-platform.en.git $(REPO)
	@for d in rtcdp xdm identity-service profile segmentation destinations sources data-governance privacy-service; do \
		if [ -d "$(REPO)/help/$$d" ]; then \
			aws s3 sync "$(REPO)/help/$$d" "s3://$$BUCKET/corpus/$$d/" \
				--exclude "*" --include "*.md" --only-show-errors --profile $(PROFILE) ; \
			echo "synced: $$d" ; \
		else \
			echo "skipping missing directory: $$d" ; \
		fi ; \
	done
	@aws s3 ls "s3://$$BUCKET/corpus/" --recursive --profile $(PROFILE) | grep '\.md$$' | wc -l

eval-set:
	$(PY) scripts/build_eval_set.py --repo $(REPO)

test:
	EMBEDDING_PROVIDER=null $(PYTEST) -q

lint:
	$(RUFF) check .
	$(RUFF) format --check .

fmt:
	$(RUFF) format .

clean:
	rm -rf .pytest_cache .ruff_cache cdk.out
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# --- stages 2 through 5 land in later commits -------------------------------
build deploy ingest local-ingest ask local-ask bench citations destroy:
	@echo "'$@' is not implemented yet. It arrives in a later stage; see README."
	@exit 1

eval:
	@echo "'eval' arrives with scripts/evaluate.py in stage 2."
	@exit 1
