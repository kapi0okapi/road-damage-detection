# Deploying the API to AWS App Runner

App Runner runs a container behind HTTPS with autoscaling and needs no cluster or load
balancer of your own. Small CPU instances are enough for ONNX Runtime at a few requests per
second.

1. `rdd-export --weights <best.pt> --out models/road_damage.onnx`
2. `AWS_REGION=eu-north-1 bash deploy/aws/create_service.sh`: creates the ECR repository and
   the App Runner access role, pushes the first image, creates the service and prints its URL.
3. For later releases, attach `road_damage.onnx` to a GitHub release and run the **deploy-aws**
   workflow (Actions → deploy-aws → Run workflow). It pushes the image to ECR and updates the
   service. It authenticates with GitHub OIDC, so no AWS keys are stored in GitHub.
4. Test it:

   ```bash
   curl https://<service-url>/health
   curl -F image=@road.jpg "https://<service-url>/detect?tiles=2"
   ```

Pause the service when it is not in use (`aws apprunner pause-service --service-arn ...`) to
keep costs at zero.
