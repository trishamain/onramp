"""Onramp infrastructure.

Stage 2 provisioned storage and messaging; stage 3 adds the container Lambdas.
Splitting it that way meant retrieval quality could be measured against a real
table before anyone waited on a 10-minute container build -- which is how the
chunk-size/model mismatch was caught before it was baked into an image.

Every resource carries a one-sentence comment explaining what it does, because
this is also a teaching artifact.
"""

from __future__ import annotations

import json

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as lambda_events
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_sqs as sqs
from aws_cdk import custom_resources as cr
from constructs import Construct

# The queue's visibility timeout must be a multiple of this, so the two are
# declared together rather than drifting apart in separate blocks.
INGEST_LAMBDA_TIMEOUT_SECONDS = 300

# Ingest loads a 130 MB model and embeds batches; 2048 MB also buys proportional
# vCPU on Lambda, which is what actually makes the embedding fast.
INGEST_MEMORY_MB = 2048
QUERY_MEMORY_MB = 2048


class OnrampStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Tag every resource in the stack so cost allocation and cleanup can
        # both filter on one key.
        Tags.of(self).add("Project", "onramp")

        # --- Corpus bucket --------------------------------------------------
        # S3 holds the Adobe markdown. Object storage rather than a database
        # because the source of truth is files, and S3 emits events on write,
        # which is what makes ingest event-driven rather than scheduled.
        self.corpus_bucket = s3.Bucket(
            self,
            "CorpusBucket",
            # Versioning keeps the previous copy of an overwritten doc, so a bad
            # sync is recoverable without re-cloning upstream.
            versioned=True,
            # Encrypt at rest with S3-managed keys. SSE-S3 rather than KMS: no
            # per-request key cost and no key policy to get wrong, and the
            # content is public documentation, not secrets.
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            # DESTROY + auto_delete so `make destroy` leaves nothing billable.
            # Safe here precisely because the corpus is reproducible from a
            # public git repo; it would be wrong for a bucket holding originals.
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- Dead letter queue ----------------------------------------------
        # Messages that fail processing three times land here instead of
        # retrying forever, so one malformed document cannot saturate the
        # ingest Lambda's concurrency indefinitely.
        self.dead_letter_queue = sqs.Queue(
            self,
            "IngestDLQ",
            retention_period=Duration.days(14),
            enforce_ssl=True,
        )

        # --- Ingest queue ---------------------------------------------------
        # SQS buffers S3 events so a bulk sync of 1,100 objects becomes a
        # drainable backlog rather than 1,100 simultaneous Lambda invocations.
        # It also gives retries and a DLQ, which a direct S3->Lambda wiring
        # does not. See README tradeoffs.
        self.ingest_queue = sqs.Queue(
            self,
            "IngestQueue",
            # Visibility timeout must exceed the consumer's timeout or SQS will
            # redeliver a message that is still being processed, producing
            # duplicate work. 6x is the AWS-recommended headroom.
            visibility_timeout=Duration.seconds(INGEST_LAMBDA_TIMEOUT_SECONDS * 6),
            retention_period=Duration.days(4),
            enforce_ssl=True,
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=3,
                queue=self.dead_letter_queue,
            ),
        )

        # Wire S3 object creation to the queue. CDK adds the queue policy that
        # lets S3 publish, which is the manual console step this replaces.
        self.corpus_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(self.ingest_queue),
            s3.NotificationKeyFilter(prefix="corpus/", suffix=".md"),
        )

        # --- Chunk table ----------------------------------------------------
        # DynamoDB stores one item per chunk, including its embedding as packed
        # float32 Binary. Single-digit-millisecond reads and no capacity to
        # manage; the tradeoff is that similarity search is a full scan, which
        # is fine at ~1,100 chunks. See README for where that stops being true.
        self.table = dynamodb.Table(
            self,
            "ChunkTable",
            partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            # On-demand: ingest is bursty and queries are sporadic, so paying
            # per request beats provisioning for a peak that is mostly idle.
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            # Point-in-time recovery gives 35 days of second-granularity
            # restore. Cheap insurance against a bad re-ingest.
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )


        # --- Provider selection ---------------------------------------------
        # ONE context value drives which embedding model runs, what the Lambdas
        # get in their environment, and whether Bedrock IAM is granted at all.
        # `cdk deploy -c embeddingProvider=bedrock` is the entire swap.
        embedding_provider = self.node.try_get_context("embeddingProvider") or "local"
        enable_generation = bool(self.node.try_get_context("enableGeneration"))

        common_env = {
            "TABLE_NAME": self.table.table_name,
            "CORPUS_BUCKET": self.corpus_bucket.bucket_name,
            "EMBEDDING_PROVIDER": embedding_provider,
            "ENABLE_GENERATION": str(enable_generation).lower(),
            # Retrieval tuning is passed through as environment rather than left
            # to the code default. Without this, changing the threshold means a
            # ~5 minute image rebuild and push; as an env var it is a config
            # update measured in seconds. The value itself is calibrated -- see
            # config.py and the README threshold table.
            "SIMILARITY_THRESHOLD": str(self.node.try_get_context("similarityThreshold") or "0.62"),
            "TOP_K": str(self.node.try_get_context("topK") or "4"),
        }

        # --- Ingest Lambda ---------------------------------------------------
        # A container image because the embedding model plus torch is far past
        # the 250 MB limit for a zip-packaged Lambda. arm64 matches the base
        # image and is cheaper per GB-second than x86.
        # Explicit log groups rather than the deprecated logRetention prop.
        # 7-day retention: long enough to debug a bad ingest, short enough that
        # CloudWatch storage never becomes a line item.
        # Functions get explicit names so their log groups can live at the
        # conventional /aws/lambda/<name> path. With CDK-generated names the
        # log group ends up somewhere like OnrampStack-IngestLogGroup131E2484,
        # and `aws logs tail /aws/lambda/<fn>` -- the command everyone reaches
        # for -- silently finds nothing.
        ingest_fn_name = "onramp-ingest"
        query_fn_name = "onramp-query"

        ingest_logs = logs.LogGroup(
            self,
            "IngestLogGroup",
            log_group_name=f"/aws/lambda/{ingest_fn_name}",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )
        query_logs = logs.LogGroup(
            self,
            "QueryLogGroup",
            log_group_name=f"/aws/lambda/{query_fn_name}",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.ingest_fn = lambda_.DockerImageFunction(
            self,
            "IngestFunction",
            function_name=ingest_fn_name,
            code=lambda_.DockerImageCode.from_image_asset(
                directory=".",
                file="docker/Dockerfile.ingest",
                platform=ecr_assets.Platform.LINUX_ARM64,
            ),
            architecture=lambda_.Architecture.ARM_64,
            memory_size=INGEST_MEMORY_MB,
            timeout=Duration.seconds(INGEST_LAMBDA_TIMEOUT_SECONDS),
            environment=common_env,
            # X-Ray active tracing: shows where a slow invocation actually went,
            # which for a container Lambda is usually init rather than handler.
            tracing=lambda_.Tracing.ACTIVE,
            log_group=ingest_logs,
        )

        # SQS as the event source rather than S3 invoking Lambda directly: this
        # is what gives batching, retries, and a DLQ. Batch size 5 keeps a single
        # failure's blast radius small; partial batch responses shrink it to one.
        self.ingest_fn.add_event_source(
            lambda_events.SqsEventSource(
                self.ingest_queue,
                batch_size=5,
                # Without this, one bad document fails the whole batch of five
                # and SQS redelivers all of them -- re-ingesting four documents
                # that already succeeded.
                report_batch_item_failures=True,
            )
        )

        # --- Query Lambda ----------------------------------------------------
        # Same image recipe, different entrypoint. 2048 MB is about vCPU as much
        # as memory: the vector matrix is ~10 MB, but numpy scoring and model
        # load both scale with the CPU share that memory buys.
        self.query_fn = lambda_.DockerImageFunction(
            self,
            "QueryFunction",
            function_name=query_fn_name,
            code=lambda_.DockerImageCode.from_image_asset(
                directory=".",
                file="docker/Dockerfile.query",
                platform=ecr_assets.Platform.LINUX_ARM64,
            ),
            architecture=lambda_.Architecture.ARM_64,
            memory_size=QUERY_MEMORY_MB,
            timeout=Duration.seconds(30),
            environment=common_env,
            tracing=lambda_.Tracing.ACTIVE,
            log_group=query_logs,
        )

        # --- IAM: least privilege, resource-scoped ---------------------------
        # CDK's grant_* helpers scope to this exact bucket/table ARN rather than
        # a wildcard, and emit only the actions each grant names.
        self.corpus_bucket.grant_read(self.ingest_fn)
        self.table.grant_read_write_data(self.ingest_fn)
        self.table.grant_read_data(self.query_fn)

        # Bedrock is granted ONLY when the same context value selects it. The
        # default deploy uses the local model, so nothing here can call Bedrock
        # even if code tried to -- which also means the quota defect on this
        # account cannot affect the deployed system.
        if embedding_provider == "bedrock" or enable_generation:
            self.ingest_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["bedrock:InvokeModel"],
                    # Scoped to foundation models and inference profiles in this
                    # account/region, not "*".
                    resources=[
                        f"arn:aws:bedrock:{self.region}::foundation-model/*",
                        f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/*",
                    ],
                )
            )
            self.query_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["bedrock:InvokeModel"],
                    resources=[
                        f"arn:aws:bedrock:{self.region}::foundation-model/*",
                        f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/*",
                    ],
                )
            )



        # --- API Gateway ------------------------------------------------------
        # A REST API rather than the cheaper HTTP API specifically for usage
        # plans and API keys: HTTP APIs have no native key/quota mechanism, and
        # throttling a pre-sales demo tool is the point. See README tradeoffs.
        self.api = apigw.RestApi(
            self,
            "OnrampApi",
            rest_api_name="onramp",
            description="Retrieval over Adobe Real-Time CDP documentation",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                # Trace every request end to end alongside the Lambda's X-Ray
                # segments, so a slow call can be attributed to API Gateway,
                # Lambda init, or the handler.
                tracing_enabled=True,
                metrics_enabled=True,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=["POST", "OPTIONS"],
                allow_headers=["Content-Type", "x-api-key"],
            ),
        )

        ask = self.api.root.add_resource("ask")
        ask.add_method(
            "POST",
            apigw.LambdaIntegration(self.query_fn, proxy=True),
            # A key is required so the endpoint cannot be scraped anonymously,
            # and so the usage plan below has something to meter against.
            api_key_required=True,
        )

        # The key itself. Its value is generated by API Gateway and retrieved
        # with the CLI command in the stack outputs -- never printed into
        # CloudFormation output, which would put a credential in plain text.
        api_key = self.api.add_api_key("OnrampApiKey", description="Onramp /ask key")

        self.api.add_usage_plan(
            "OnrampUsagePlan",
            name="onramp-default",
            api_stages=[apigw.UsagePlanPerApiStage(api=self.api, stage=self.api.deployment_stage)],
            # 5 rps sustained, 10 request burst. Sized for a demo tool: enough
            # for a live conversation, low enough that a loop in someone's
            # script cannot run up a bill against a 1.87 GB container Lambda.
            throttle=apigw.ThrottleSettings(rate_limit=5, burst_limit=10),
        ).add_api_key(api_key)


        # --- Demo page --------------------------------------------------------
        # A zip Lambda, not the query container: serving static HTML from a
        # 1.87 GB image would mean a 12.5 s cold start to render a page. This
        # one imports only the standard library and starts in well under a
        # second. See web/web_handler.py for why this is an API Gateway route
        # rather than S3 static website hosting.
        web_logs = logs.LogGroup(
            self,
            "WebLogGroup",
            log_group_name="/aws/lambda/onramp-web",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.web_fn = lambda_.Function(
            self,
            "WebFunction",
            function_name="onramp-web",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="web_handler.handler",
            code=lambda_.Code.from_asset("web"),
            memory_size=256,
            timeout=Duration.seconds(10),
            tracing=lambda_.Tracing.ACTIVE,
            log_group=web_logs,
        )

        # GET / serves the page. No API key here -- a key on the HTML would have
        # to be embedded in the page to be usable, which is exactly what we are
        # avoiding. The page prompts for the key and holds it in memory; the
        # data route (/ask) stays key-protected and metered.
        self.api.root.add_method(
            "GET",
            apigw.LambdaIntegration(self.web_fn, proxy=True),
            api_key_required=False,
        )

        # --- ECR lifecycle ----------------------------------------------------
        # Every `cdk deploy` pushes a new ~1.9 GB image. Without a lifecycle
        # rule those accumulate at $0.10/GB-month forever. Keep the last 3 so a
        # rollback target exists without paying for every build ever made.
        #
        # CDK publishes container assets into the shared bootstrap repo
        # (cdk-hnb659fds-container-assets-<account>-<region>), so the rule is
        # applied there by reference rather than to a repo this stack owns.
        asset_repo_name = f"cdk-hnb659fds-container-assets-{self.account}-{self.region}"
        cdk_asset_repo = ecr.Repository.from_repository_name(self, "CdkAssetRepo", asset_repo_name)

        # from_repository_name returns an IRepository, which has no
        # add_lifecycle_rule -- CDK only manages rules on repos it owns. A
        # custom resource applies the policy to the imported repo so this stays
        # `cdk deploy` with zero console steps, as specified.
        #
        # CAVEAT: this is the SHARED bootstrap asset repo. Every CDK app in this
        # account publishes container assets here, so "keep 3" is account-wide,
        # not per-stack. Correct while Onramp is the only CDK app here; it would
        # need per-app repos if that changed.
        ecr_lifecycle = cr.AwsCustomResource(
            self,
            "EcrLifecyclePolicy",
            on_update=cr.AwsSdkCall(
                service="ECR",
                action="putLifecyclePolicy",
                parameters={
                    "repositoryName": asset_repo_name,
                    "lifecyclePolicyText": json.dumps(
                        {
                            "rules": [
                                {
                                    "rulePriority": 1,
                                    "description": "Keep only the last 3 images; each is ~1.9 GB",
                                    "selection": {
                                        "tagStatus": "any",
                                        "countType": "imageCountMoreThan",
                                        "countNumber": 3,
                                    },
                                    "action": {"type": "expire"},
                                }
                            ]
                        }
                    ),
                },
                physical_resource_id=cr.PhysicalResourceId.of(f"{asset_repo_name}-lifecycle"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=["ecr:PutLifecyclePolicy"],
                        resources=[
                            f"arn:aws:ecr:{self.region}:{self.account}:repository/{asset_repo_name}"
                        ],
                    )
                ]
            ),
        )
        ecr_lifecycle.node.add_dependency(self.ingest_fn)

        # --- Outputs ---------------------------------------------------------
        CfnOutput(self, "CorpusBucketName", value=self.corpus_bucket.bucket_name)
        CfnOutput(self, "ChunkTableName", value=self.table.table_name)
        CfnOutput(self, "IngestQueueUrl", value=self.ingest_queue.queue_url)
        CfnOutput(self, "DeadLetterQueueUrl", value=self.dead_letter_queue.queue_url)
        CfnOutput(self, "IngestFunctionName", value=self.ingest_fn.function_name)
        CfnOutput(self, "IngestLogGroupName", value=ingest_logs.log_group_name)
        CfnOutput(self, "QueryFunctionName", value=self.query_fn.function_name)
        CfnOutput(self, "QueryLogGroupName", value=query_logs.log_group_name)
        CfnOutput(self, "ApiEndpoint", value=f"{self.api.url}ask")
        CfnOutput(self, "DemoUiUrl", value=self.api.url, description="Open in a browser")
        CfnOutput(
            self,
            "GetApiKeyCommand",
            value=(
                f"aws apigateway get-api-key --api-key {api_key.key_id} "
                f"--include-value --region {self.region} --query value --output text"
            ),
            description="Retrieve the API key value (never stored in the template)",
        )
        CfnOutput(self, "EmbeddingProvider", value=embedding_provider)
        CfnOutput(self, "EcrAssetRepo", value=cdk_asset_repo.repository_uri)
        CfnOutput(
            self,
            "SyncCorpusCommand",
            value=f"aws s3 sync <local-repo> s3://{self.corpus_bucket.bucket_name}/corpus/",
            description="Copy the Adobe markdown into the CDK-managed bucket",
        )
