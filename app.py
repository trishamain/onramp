#!/usr/bin/env python3
"""CDK entry point.

Account and region come from the CDK environment (CDK_DEFAULT_ACCOUNT /
CDK_DEFAULT_REGION, which the CLI populates from the active AWS profile), never
from a hardcoded literal -- so this deploys into whatever account the caller is
authenticated to.
"""

import os

import aws_cdk as cdk
from infra.onramp_stack import OnrampStack

app = cdk.App()

OnrampStack(
    app,
    "OnrampStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    ),
    description="Onramp: serverless RAG over Adobe Real-Time CDP documentation",
)

app.synth()
