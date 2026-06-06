#!/bin/bash
# One-time AWS setup: ECR repository, App Runner access role, first image push and the service.
# Needs: AWS CLI v2 logged in (aws sso login / access keys), Docker, models/road_damage.onnx.
#
#   AWS_REGION=eu-north-1 bash deploy/aws/create_service.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

REGION=${AWS_REGION:-eu-north-1}
REPO=${ECR_REPOSITORY:-road-damage-api}
SERVICE=${SERVICE_NAME:-road-damage-api}
ROLE=${ROLE_NAME:-road-damage-apprunner-ecr}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$REPO:initial"

aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null

if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "build.apprunner.amazonaws.com"},
                   "Action": "sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  sleep 10  # IAM propagation
fi
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

aws apprunner create-service --region "$REGION" --service-name "$SERVICE" \
  --source-configuration "ImageRepository={ImageIdentifier=$IMAGE,ImageRepositoryType=ECR,ImageConfiguration={Port=8080}},AutoDeploymentsEnabled=false,AuthenticationConfiguration={AccessRoleArn=$ROLE_ARN}" \
  --instance-configuration "Cpu=1 vCPU,Memory=2 GB" \
  --health-check-configuration "Protocol=HTTP,Path=/health,Interval=10,Timeout=5,HealthyThreshold=1,UnhealthyThreshold=5" \
  --query "Service.[ServiceArn,ServiceUrl]" --output text

echo "ECR access role: $ROLE_ARN  (GitHub variable APPRUNNER_ECR_ROLE_ARN)"
