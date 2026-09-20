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

.PHONY: help venv load test lint fmt eval-set build deploy ingest queue local-ingest \
        ask local-ask bench eval eval-api citations destroy clean

help:
	@echo "venv         create .venv with python3.12 and install dev deps"
	@echo "load         clone the Adobe docs repo to \$$REPO and sync markdown to \$$BUCKET"
	@echo "eval-set     validate golden-draft.json and write evals/golden.json"
	@echo "test         run pytest with the Null provider"
	@echo "lint         ruff check + format --check"
	@echo "fmt          ruff format"
	@echo "build        build both Lambda container images locally"
	@echo "deploy       cdk deploy"
	@echo "ingest       re-trigger the deployed event-driven ingest"
	@echo "queue        show ingest queue + DLQ depth"
	@echo "local-ingest run the identical chunker/embedder on this laptop"
	@echo "ask          query the deployed API:  make ask Q=\"...\""
	@echo "local-ask    query the same logic locally: make local-ask Q=\"...\""
	@echo "bench        p50/p95 across four conditions"
	@echo "eval         recall/refusal against the local path"
	@echo "eval-api     the same, against the deployed API"
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

# --- stack values, read once from CloudFormation outputs ---------------------
# Queried rather than hardcoded so a fresh deploy into another account works
# with no edits. `make` expands these on every target, so they are :=-assigned
# only inside the targets that need them.
CFN = aws cloudformation describe-stacks --stack-name OnrampStack --profile $(PROFILE) --region $(REGION)
OUT = --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text

build:
	docker build --platform linux/arm64 -f docker/Dockerfile.ingest -t onramp-ingest:latest .
	docker build --platform linux/arm64 -f docker/Dockerfile.query  -t onramp-query:latest .
	@echo "--- image sizes (2 GB is the working ceiling) ---"
	@docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}" | grep onramp

deploy:
	@echo "First build takes 8-15 minutes; later ones reuse the torch layer."
	AWS_PROFILE=$(PROFILE) \
	CDK_DEFAULT_ACCOUNT=$$(aws sts get-caller-identity --profile $(PROFILE) --query Account --output text) \
	CDK_DEFAULT_REGION=$(REGION) \
	JSII_SILENCE_WARNING_DEPRECATED_NODE_VERSION=1 \
	npx --yes aws-cdk@latest deploy --require-approval never

ingest:
	@b=$$($(CFN) $(call OUT,CorpusBucketName)) ; \
	echo "re-triggering ingest by rewriting every object in s3://$$b/corpus/" ; \
	aws s3 cp "s3://$$b/corpus/" "s3://$$b/corpus/" --recursive \
	  --exclude "*" --include "*.md" --metadata-directive REPLACE --metadata rerun=make \
	  --only-show-errors --profile $(PROFILE) --region $(REGION) ; \
	echo "queued; watch with: make queue"

queue:
	@q=$$($(CFN) $(call OUT,IngestQueueUrl)) ; \
	d=$$($(CFN) $(call OUT,DeadLetterQueueUrl)) ; \
	aws sqs get-queue-attributes --queue-url $$q --attribute-names \
	  ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible \
	  --query Attributes --output json --profile $(PROFILE) --region $(REGION) ; \
	printf "DLQ: " ; aws sqs get-queue-attributes --queue-url $$d \
	  --attribute-names ApproximateNumberOfMessages \
	  --query Attributes.ApproximateNumberOfMessages --output text --profile $(PROFILE) --region $(REGION)

local-ingest:
	@t=$$($(CFN) $(call OUT,ChunkTableName)) ; \
	b=$$($(CFN) $(call OUT,CorpusBucketName)) ; \
	caffeinate -i $(PY) scripts/local_ingest.py --table $$t --bucket $$b --profile $(PROFILE)

ask:
	@test -n "$(Q)" || { echo 'usage: make ask Q="your question"'; exit 1; }
	@e=$$($(CFN) $(call OUT,ApiEndpoint)) ; \
	k=$$(aws apigateway get-api-keys --include-values --profile $(PROFILE) --region $(REGION) \
	      --query "items[0].value" --output text) ; \
	curl -s -X POST "$$e" -H "x-api-key: $$k" -H "Content-Type: application/json" \
	  -d "{\"question\":\"$(Q)\"}" | python3 -m json.tool

local-ask:
	@test -n "$(Q)" || { echo 'usage: make local-ask Q="your question"'; exit 1; }
	@t=$$($(CFN) $(call OUT,ChunkTableName)) ; \
	$(PY) scripts/local_ask.py --table $$t --profile $(PROFILE) "$(Q)"

bench:
	@e=$$($(CFN) $(call OUT,ApiEndpoint)) ; \
	t=$$($(CFN) $(call OUT,ChunkTableName)) ; \
	k=$$(aws apigateway get-api-keys --include-values --profile $(PROFILE) --region $(REGION) \
	      --query "items[0].value" --output text) ; \
	$(PY) scripts/benchmark.py --endpoint "$$e" --api-key "$$k" --table $$t --profile $(PROFILE)

eval:
	@t=$$($(CFN) $(call OUT,ChunkTableName)) ; \
	$(PY) scripts/evaluate.py --mode local --table $$t --profile $(PROFILE)

eval-api:
	@e=$$($(CFN) $(call OUT,ApiEndpoint)) ; \
	k=$$(aws apigateway get-api-keys --include-values --profile $(PROFILE) --region $(REGION) \
	      --query "items[0].value" --output text) ; \
	$(PY) scripts/evaluate.py --mode api --endpoint "$$e" --api-key "$$k"

citations:
	@t=$$($(CFN) $(call OUT,ChunkTableName)) ; \
	$(PY) scripts/verify_citations.py --table $$t --profile $(PROFILE)

destroy:
	AWS_PROFILE=$(PROFILE) \
	CDK_DEFAULT_ACCOUNT=$$(aws sts get-caller-identity --profile $(PROFILE) --query Account --output text) \
	CDK_DEFAULT_REGION=$(REGION) \
	JSII_SILENCE_WARNING_DEPRECATED_NODE_VERSION=1 \
	npx --yes aws-cdk@latest destroy --force
