# Distributed deployment configuration (as provided in the document)
deployment_config = {
    'expert_parallel_size': 8,      # Experts across 8 GPUs
    'pipeline_parallel_size': 2,    # 2-stage pipeline
    'capacity_factor_inference': 2.0,  # Higher for quality (expert capacity during inference)
    'top_k_inference': 2,           # Balance quality/speed by selecting top-k experts
    'enable_expert_caching': True,  # Whether to use LRU cache for experts
    'cache_size_gb': 16,            # Size of the LRU cache in GB
    'batch_timeout_ms': 50,         # Timeout for batching requests in ms
    'max_batch_size': 32            # Maximum batch size for inference server
}

if __name__ == '__main__':
    print("Deployment Configuration:")
    for key, value in deployment_config.items():
        print(f"  {key}: {value}")
