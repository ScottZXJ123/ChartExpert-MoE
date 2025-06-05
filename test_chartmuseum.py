#!/usr/bin/env python3

from datasets import load_dataset

print('Loading ChartMuseum dataset...')
ds = load_dataset("lytang/ChartMuseum")
print(f'✅ Success! Dataset loaded with splits: {list(ds.keys())}')

for split_name, split_data in ds.items():
    print(f'{split_name}: {len(split_data)} samples')

# Test accessing a sample
test_sample = ds['test'][0]
print(f'\nSample keys: {list(test_sample.keys())}')
print(f'Question: {test_sample.get("question", "N/A")}')
print(f'Answer: {test_sample.get("answer", "N/A")}')

# Show image info if available
if 'image' in test_sample:
    image = test_sample['image']
    print(f'Image type: {type(image)}')
    if hasattr(image, 'size'):
        print(f'Image size: {image.size}') 