# ChartExpert-MoE GPU Performance Report

## GPU Configuration
- **Device**: NVIDIA A40
- **Memory**: 44.4 GB
- **Compute Capability**: 8.6
- **PyTorch Version**: 2.5.1+cu121
- **CUDA Version**: 12.1

## Performance Summary

### Memory Efficiency
- **Peak Memory Usage**: 1.25 GB
- **Average Memory Usage**: 0.16 GB
- **Memory Efficiency**: 0.4%

### Performance Metrics

- **Alloc 1024X1024**: 2.21ms
- **Alloc 2048X2048**: 1.33ms
- **Alloc 4096X4096**: 1.64ms
- **Alloc 8192X4096**: 7.06ms
- **Attention 1024 16**: 146.02ms
- **Attention 1024 32**: 289.95ms
- **Attention 1024 8**: 74.88ms
- **Attention 128 16**: 12.73ms
- **Attention 128 32**: 23.08ms
- **Attention 128 8**: 5.68ms
- **Attention 256 16**: 22.10ms
- **Attention 256 32**: 42.60ms
- **Attention 256 8**: 11.46ms
- **Attention 512 16**: 52.71ms
- **Attention 512 32**: 103.76ms
- **Attention 512 8**: 27.54ms
- **Conv 224 16**: 174.90ms
- **Conv 224 32**: 343.87ms
- **Conv 224 64**: 682.57ms
- **Conv 224 8**: 90.61ms
- **Conv 384 16**: 521.14ms
- **Conv 384 32**: 1037.32ms
- **Conv 384 64**: 2074.83ms
- **Conv 384 8**: 271.09ms
- **Conv 512 16**: 917.99ms
- **Conv 512 32**: 1835.13ms
- **Conv 512 64**: 5991.26ms
- **Conv 512 8**: 478.97ms
- **Expert Exec 16**: 14.35ms
- **Expert Exec 32**: 14.23ms
- **Expert Exec 64**: 8.49ms
- **Free 1024X1024**: 0.18ms
- **Free 2048X2048**: 3.91ms
- **Free 4096X4096**: 9.23ms
- **Free 8192X4096**: 10.88ms
- **Routing 16**: 58.35ms
- **Routing 32**: 3.25ms
- **Routing 64**: 3.03ms
- **Simplified 128**: 2.47ms
- **Simplified 16**: 1.98ms
- **Simplified 32**: 2.11ms
- **Simplified 64**: 2.17ms

## Performance Grades
- **Memory Efficiency**: A
- **Speed Performance**: C

## Recommendations

### Memory Optimization
- Memory usage is within acceptable range
- Consider increasing batch sizes for better throughput

### Speed Optimization
- Enable mixed precision training (FP16)
- Optimize data loading pipeline

---
*Report generated: 2025-06-05 21:43:39*
