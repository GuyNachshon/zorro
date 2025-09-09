# Training Pipeline Fix

## Problem Identified

The ICN training script (`train_icn.py`) was hanging due to:

1. **Parallel Processing Issue**: The `ICNDataPreparator.prepare_complete_dataset()` method uses `ProcessPoolExecutor` to process samples in parallel
2. **CPU Overload**: Multiple Python processes spawned by the executor were consuming 99.9% CPU each
3. **Missing Error Handling**: No timeout or proper error handling for the parallel processing
4. **Dataset Issues**: The malicious dataset might not exist or be properly formatted

## Immediate Solution

### 1. Kill Hanging Processes

```bash
# Make the kill script executable
chmod +x kill_training.sh

# Kill the hanging processes
./kill_training.sh
```

Or manually:
```bash
pkill -f "train_icn.py"
pkill -f "train_all.py"
```

### 2. Use Fixed Training Scripts

We've created fixed versions that avoid the hanging issue:

#### Option A: Fixed ICN Training Only
```bash
# Run the fixed ICN training script
python icn_training_fix.py
```

#### Option B: Complete Fixed Pipeline
```bash
# Run all models with timeouts and error handling
python train_all_fixed.py --models icn amil cpg neobert --fix-icn --timeout 300

# Or use minimal scripts for quick testing
python train_all_fixed.py --use-minimal --models amil cpg neobert
```

## Root Cause Analysis

The hanging occurs in `icn/data/data_preparation.py` at lines 285-321:

```python
# This code spawns multiple processes that can hang
with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
    future_to_sample = {
        executor.submit(self.process_single_malicious_sample, sample): sample
        for sample in all_malicious
    }
```

Issues:
1. No timeout on the executor tasks
2. If `process_single_malicious_sample` fails or hangs, the entire pool waits indefinitely
3. The malicious dataset extraction might be stuck on I/O operations

## Permanent Fix

To permanently fix the ICN training, modify `icn/data/data_preparation.py`:

### Option 1: Disable Parallel Processing (Quick Fix)

Replace the ProcessPoolExecutor sections with sequential processing:

```python
# Instead of ProcessPoolExecutor, use sequential processing
successful_malicious = 0
for sample in all_malicious:
    try:
        processed = self.process_single_malicious_sample(sample)
        if processed:
            processed_packages.append(processed)
            successful_malicious += 1
            
        if successful_malicious % 10 == 0:
            logger.info(f"✅ Processed {successful_malicious}/{len(all_malicious)} malicious samples")
    except Exception as e:
        logger.warning(f"Failed to process {sample.name}: {e}")
        continue
```

### Option 2: Add Timeout to Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

# Add timeout to futures
with ProcessPoolExecutor(max_workers=min(4, self.parallel_workers)) as executor:
    futures = {
        executor.submit(self.process_single_malicious_sample, sample): sample
        for sample in all_malicious
    }
    
    for future in as_completed(futures, timeout=300):  # 5 minute total timeout
        try:
            processed = future.result(timeout=10)  # 10 second per-task timeout
            if processed:
                processed_packages.append(processed)
                successful_malicious += 1
        except TimeoutError:
            logger.warning(f"Sample processing timed out")
        except Exception as e:
            logger.warning(f"Sample processing failed: {e}")
```

## Training Recommendations

1. **Start Small**: Begin with minimal datasets to verify everything works
   ```bash
   python train_all_fixed.py --use-minimal --max-epochs 2
   ```

2. **Use Timeouts**: Always set reasonable timeouts
   ```bash
   python train_all_fixed.py --timeout 120 --models icn amil
   ```

3. **Monitor Resources**: Watch CPU and memory usage
   ```bash
   # In another terminal
   watch -n 1 'ps aux | grep python | grep -v grep'
   ```

4. **Check Logs**: Monitor training progress
   ```bash
   tail -f logs/train_all_fixed.log
   ```

## Features of Fixed Training Pipeline

The `train_all_fixed.py` script includes:

- **Timeout Protection**: Each model training has a configurable timeout
- **Graceful Termination**: Properly kills hanging processes
- **Error Recovery**: Continue training other models even if one fails
- **Minimal Mode**: Use synthetic data for quick testing
- **Progress Logging**: Clear indication of what's happening
- **Resource Management**: Prevents CPU overload

## Next Steps

After fixing the immediate issue:

1. Test with minimal data first
2. Gradually increase dataset size
3. Monitor system resources
4. Consider using GPU if available for faster training
5. Implement proper checkpointing for long training runs