# AWS Deployment Guide for CSM Demo

This guide outlines how to deploy a web-based demo of the Conversational Speech Model (CSM) on AWS, similar to Sesame's voice demo.

## Architecture Overview

```
                                    ┌─────────────────┐
                                    │   CloudFront    │
                                    │   Distribution  │
                                    └────────┬────────┘
                                            │
                                    ┌───────▼────────┐
                                    │      S3        │
                                    │  Static Site   │
                                    └───────┬────────┘
                                            │
┌─────────────┐    WebSocket     ┌─────────▼────────┐
│   Browser   │◄───Connection────►│   API Gateway    │
└─────────────┘                   └───────┬─────────┘
                                          │
                                  ┌───────▼────────┐
                                  │    Lambda      │
                                  │   Function     │
                                  └───────┬────────┘
                                          │
                                  ┌───────▼────────┐
                                  │      ECS       │
                                  │    Service     │
                                  └───────┬────────┘
                                          │
                                  ┌───────▼────────┐
                                  │      ECR       │
                                  │   Registry     │
                                  └───────┬────────┘
                                          │
                                  ┌───────▼────────┐
                                  │      EFS       │
                                  │    Storage     │
                                  └───────────────┘

```

## Components

1. **Frontend (S3 + CloudFront)**
   - React/TypeScript web application
   - WebSocket client for real-time audio streaming
   - Audio recording and playback capabilities
   - Responsive UI matching Sesame's demo style

2. **API Layer (API Gateway + Lambda)**
   - WebSocket API for bidirectional communication
   - Lambda functions for connection management
   - Audio streaming protocol implementation
   - Error handling and retry logic

3. **Processing Layer (ECS + ECR)**
   - Containerized CSM service
   - GPU-optimized instances (g4dn.xlarge)
   - Auto-scaling configuration
   - Health monitoring

4. **Storage Layer (EFS)**
   - Model weights storage
   - Conversation history
   - Generated audio cache

## Implementation Steps

### 1. Containerize CSM

```dockerfile
FROM rust:1.70 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libssl-dev \
    pkg-config

# Copy source code
COPY . /usr/src/csm
WORKDIR /usr/src/csm

# Build release
RUN cargo build --release

# Runtime image
FROM nvidia/cuda:12.0.0-base-ubuntu20.04

# Copy binary and resources
COPY --from=builder /usr/src/csm/target/release/csm /usr/local/bin/
COPY models/ /models/

# Set up environment
ENV RUST_LOG=info
ENV MODEL_PATH=/models/llama-1B.pth
ENV MODEL_FLAVOR=llama-1B

# Run service
CMD ["csm", "--server"]
```

### 2. Set Up Infrastructure (Terraform)

```hcl
# ECS Cluster
resource "aws_ecs_cluster" "csm_cluster" {
  name = "csm-cluster"
  capacity_providers = ["FARGATE_SPOT"]
}

# ECS Service
resource "aws_ecs_service" "csm_service" {
  name            = "csm-service"
  cluster         = aws_ecs_cluster.csm_cluster.id
  task_definition = aws_ecs_task_definition.csm_task.arn
  desired_count   = 2

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [aws_security_group.csm_sg.id]
  }
}

# API Gateway WebSocket API
resource "aws_apigatewayv2_api" "csm_ws" {
  name                       = "csm-websocket-api"
  protocol_type             = "WEBSOCKET"
  route_selection_expression = "$request.body.action"
}
```

### 3. Frontend Implementation

```typescript
// WebSocket client setup
const ws = new WebSocket(WS_ENDPOINT);

// Audio streaming
const startConversation = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  
  recorder.ondataavailable = (e) => {
    ws.send(JSON.stringify({
      action: 'audio',
      data: e.data
    }));
  };
  
  recorder.start(100); // Stream in 100ms chunks
};

// Handle responses
ws.onmessage = async (event) => {
  const response = JSON.parse(event.data);
  if (response.type === 'audio') {
    await playAudio(response.data);
  }
};
```

### 4. Lambda Handler

```python
import json
import boto3

def handle_connection(event, context):
    connection_id = event['requestContext']['connectionId']
    
    if event['requestContext']['eventType'] == 'CONNECT':
        # Initialize session
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Connected'})
        }
    
    if event['requestContext']['eventType'] == 'MESSAGE':
        # Process audio chunk
        data = json.loads(event['body'])
        task = {
            'connection_id': connection_id,
            'audio_data': data['data']
        }
        
        # Send to ECS for processing
        response = invoke_ecs_task(task)
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
```

## Deployment

1. **Build and Push Container**
```bash
aws ecr create-repository --repository-name csm-demo
docker build -t csm-demo .
docker tag csm-demo:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/csm-demo:latest
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/csm-demo:latest
```

2. **Deploy Infrastructure**
```bash
terraform init
terraform plan
terraform apply
```

3. **Configure DNS and SSL**
```bash
aws acm request-certificate --domain-name demo.yourdomain.com
aws cloudfront create-distribution --origin-domain-name your-s3-bucket.s3.amazonaws.com
```

## Monitoring and Scaling

1. **CloudWatch Metrics**
   - CPU/Memory utilization
   - WebSocket connection count
   - Audio processing latency
   - Error rates

2. **Auto-scaling Policies**
```hcl
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.csm_cluster.name}/${aws_ecs_service.csm_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "cpu-auto-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 75.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
```

## Cost Optimization

1. **Instance Selection**
   - Use Spot instances for cost savings
   - Right-size GPU instances
   - Implement proper auto-scaling

2. **Storage Management**
   - Cache frequently used responses
   - Implement TTL for conversation history
   - Use S3 lifecycle policies

## Security Considerations

1. **Network Security**
   - VPC configuration
   - Security groups
   - WAF rules

2. **Authentication**
   - Cognito integration
   - API key management
   - Rate limiting

3. **Data Protection**
   - Audio encryption
   - Secure WebSocket connections
   - Access logging

## Maintenance

1. **Updates**
   - Regular model updates
   - Security patches
   - Dependency updates

2. **Backup**
   - Model weights backup
   - Configuration backup
   - Database backup

3. **Monitoring**
   - Set up alerts
   - Performance monitoring
   - Error tracking

## Troubleshooting

1. **Common Issues**
   - Connection timeouts
   - Audio processing errors
   - Scaling issues

2. **Debugging**
   - CloudWatch Logs
   - X-Ray tracing
   - Error reporting

## Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [API Gateway WebSocket API](https://docs.aws.amazon.com/apigateway/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/) 