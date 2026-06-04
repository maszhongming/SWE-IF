# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Concurrent processor for LLM-based tasks.
Provides thread-safe, rate-limited concurrent processing capabilities.
"""

import threading
import time
import json
from typing import List, Dict, Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging

from .model_configs import get_concurrent_config


class ConcurrentProcessor:
    """
    Generic concurrent processor for LLM tasks with rate limiting and thread safety.
    """
    
    def __init__(self, model_name: str, enable_logging: bool = True):
        """
        Initialize concurrent processor.
        
        Args:
            model_name: Name of the model (used to get concurrent config)
            enable_logging: Whether to enable detailed logging
            
        Note:
            The concurrent_max_retries from config controls task-level retries for unexpected errors:
            - Thread failures
            - Data processing errors  
            - Unexpected exceptions
            This is different from LLM-level retries which are handled within individual processing functions.
        """
        self.model_name = model_name
        self.config = get_concurrent_config(model_name)
        
        # Concurrent settings from model config with defaults applied
        self.max_workers = self.config['max_workers']
        self.rate_limit_delay = self.config['rate_limit_delay']
        self.retry_delay = self.config['retry_delay']
        self.max_retries = self.config['concurrent_max_retries']  # For task-level unexpected errors only
        
        # Thread synchronization primitives
        self._write_lock = threading.Lock()
        self._rate_limit_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._last_request_time = 0
        
        # Progress tracking
        self._completed = 0
        self._failed = 0
        self._total = 0
        
        # Setup logging
        self.logger = self._setup_logger() if enable_logging else None
        
        if self.logger:
            self.logger.info(f"Initialized concurrent processor for {model_name}")
            self.logger.info(f"Config: max_workers={self.max_workers}, "
                           f"rate_limit={self.rate_limit_delay}s, "
                           f"concurrent_max_retries={self.max_retries} (for unexpected errors)")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup thread-safe logger with model-specific name."""
        logger = logging.getLogger(f'ConcurrentProcessor-{self.model_name}')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add console handler with thread info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _apply_rate_limit(self):
        """Apply rate limiting between requests to avoid API throttling."""
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                time.sleep(sleep_time)
            self._last_request_time = time.time()
    
    def _update_progress(self, success: bool):
        """Update progress counters in a thread-safe manner."""
        with self._progress_lock:
            if success:
                self._completed += 1
            else:
                self._failed += 1
            
            # Log progress every 10% or every 10 items, whichever is smaller
            progress_interval = max(1, min(10, self._total // 10))
            processed_count = self._completed + self._failed

            if (self._completed + self._failed) % progress_interval == 0:
                processed_pct = (processed_count / self._total) * 100 if self._total > 0 else 0
                if self.logger:
                    self.logger.info(f"Progress: {self._completed + self._failed}/{self._total} "
                                   f"({processed_pct:.1f}%) - Success: {self._completed}, Failed: {self._failed}")
    
    def _safe_file_write(self, file_handle, content: str):
        """Thread-safe file writing with immediate flush."""
        with self._write_lock:
            file_handle.write(content)
            file_handle.flush()
    
    def _process_single_item_with_retries(
        self, 
        item: Any, 
        process_func: Callable,
        item_id: str = None,
        check_result_status: bool = True
    ) -> Tuple[Any, bool, str]:
        """
        Process single item with retry logic and rate limiting.
        
        Args:
            item: Item to process
            process_func: Function to process the item
            item_id: Optional identifier for logging
            check_result_status: Whether to check result['status'] for failure detection
            
        Returns:
            Tuple of (result, success, error_message)
        """
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting before each request
                self._apply_rate_limit()
                
                # Process the item
                result = process_func(item)
                
                # If check_result_status is True, examine the result to determine success
                if check_result_status and isinstance(result, dict):
                    result_status = result.get('status', '')
                    if result_status in ['failed_after_retries', 'processing_error']:
                        # This result indicates failure, but it's a valid result
                        # Don't retry - return as successful processing with failed content
                        return result, False, f"Result indicates failure: {result_status}"
                
                # Successful processing
                return result, True, ""
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)[:150]}"
                
                if self.logger:
                    if item_id:
                        self.logger.warning(f"Item {item_id}: {error_msg}")
                    else:
                        self.logger.warning(error_msg)
                
                if attempt < self.max_retries - 1:
                    # Wait before retry with exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    # Final failure after all retries
                    final_error = f"Failed after {self.max_retries} attempts. Last error: {str(e)}"
                    return None, False, final_error
        
        return None, False, "Unknown error"
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        get_item_id: Optional[Callable[[Any], str]] = None,
        progress_desc: str = "Processing"
    ) -> List[Tuple[Any, bool, str]]:
        """
        Process a batch of items concurrently while maintaining order.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            get_item_id: Optional function to extract item ID for logging
            progress_desc: Description for progress logging
            
        Returns:
            List of (result, success, error_message) tuples in original order
        """
        if not items:
            return []
        
        self._total = len(items)
        self._completed = 0
        self._failed = 0
        
        if self.logger:
            self.logger.info(f"Starting {progress_desc} with {len(items)} items using {self.max_workers} workers")
        
        results = [None] * len(items)  # Maintain original order
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with their indices to maintain order
            future_to_index = {}
            for i, item in enumerate(items):
                future = executor.submit(
                    self._process_single_item_with_retries, 
                    item, 
                    process_func,
                    get_item_id(item) if get_item_id else f"item_{i}"
                )
                future_to_index[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result, success, error = future.result()
                    results[index] = (result, success, error)
                    self._update_progress(success)
                except Exception as e:
                    # This shouldn't happen as we handle exceptions in _process_single_item_with_retries
                    results[index] = (None, False, f"Unexpected error: {str(e)}")
                    self._update_progress(False)
        
        if self.logger:
            self.logger.info(f"{progress_desc} completed: {self._completed} succeeded, {self._failed} failed")
        
        return results
    
    def process_and_save_jsonl(
        self,
        items: List[Dict[str, Any]],
        process_func: Callable,
        output_file: Path,
        get_item_id: Optional[Callable[[Dict], str]] = None,
        progress_desc: str = "Processing",
        check_result_status: bool = True
    ) -> int:
        """
        Process items and save results to JSONL file maintaining order.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            output_file: Path to output JSONL file
            get_item_id: Optional function to extract item ID
            progress_desc: Description for progress logging
            check_result_status: Whether to check result['status'] for failure detection
            
        Returns:
            Number of successfully processed items
        """
        if not items:
            return 0
        
        # Process all items concurrently
        results = self.process_batch(items, process_func, get_item_id, progress_desc, check_result_status)
        
        # Write results to file in original order
        successful_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (result, success, error) in enumerate(results):
                if success and result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    successful_count += 1
                else:
                    # Write failed item with error info
                    if result is not None:
                        # We have a result but it indicates failure
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    else:
                        # No result due to unexpected error
                        failed_item = items[i].copy() if isinstance(items[i], dict) else {"original_item": items[i]}
                        failed_item["processing_error"] = error
                        failed_item["status"] = "processing_error"
                        f.write(json.dumps(failed_item, ensure_ascii=False) + '\n')
        
        if self.logger:
            self.logger.info(f"Saved {successful_count}/{len(items)} successful results to {output_file}")
        
        return successful_count
    
    def process_and_append_jsonl(
        self,
        items: List[Dict[str, Any]],
        process_func: Callable,
        output_file: Path,
        get_item_id: Optional[Callable[[Dict], str]] = None,
        progress_desc: str = "Processing",
        check_result_status: bool = True
    ) -> int:
        """
        Process items and append results to JSONL file as they complete (for resumable processing).
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            output_file: Path to output JSONL file
            get_item_id: Optional function to extract item ID
            progress_desc: Description for progress logging
            check_result_status: Whether to check result['status'] for failure detection
            
        Returns:
            Number of successfully processed items
        """
        if not items:
            return 0
        
        self._total = len(items)
        self._completed = 0
        self._failed = 0
        
        if self.logger:
            self.logger.info(f"Starting {progress_desc} with {len(items)} items, appending to {output_file}")
        
        successful_count = 0
        
        # Open file for appending and process with immediate writing
        with open(output_file, 'a', encoding='utf-8') as f:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_item = {}
                for item in items:
                    future = executor.submit(
                        self._process_single_item_with_retries,
                        item,
                        process_func,
                        get_item_id(item) if get_item_id else str(item.get('task_id', 'unknown')),
                        check_result_status
                    )
                    future_to_item[future] = item
                
                # Process results as they complete (order may vary)
                for future in as_completed(future_to_item):
                    original_item = future_to_item[future]
                    
                    try:
                        result, success, error = future.result()
                        
                        if success and result is not None:
                            self._safe_file_write(f, json.dumps(result, ensure_ascii=False) + '\n')
                            successful_count += 1
                        else:
                            # Handle failure cases
                            if result is not None:
                                # We have a result but it indicates failure (e.g., LLM failed)
                                self._safe_file_write(f, json.dumps(result, ensure_ascii=False) + '\n')
                            else:
                                # No result due to unexpected error
                                failed_item = original_item.copy() if isinstance(original_item, dict) else {"original_item": original_item}
                                failed_item["processing_error"] = error
                                failed_item["status"] = "processing_error"
                                self._safe_file_write(f, json.dumps(failed_item, ensure_ascii=False) + '\n')
                        
                        self._update_progress(success)
                        
                    except Exception as e:
                        # Handle unexpected errors
                        failed_item = original_item.copy() if isinstance(original_item, dict) else {"original_item": original_item}
                        failed_item["processing_error"] = f"Unexpected error: {str(e)}"
                        failed_item["status"] = "processing_error"
                        self._safe_file_write(f, json.dumps(failed_item, ensure_ascii=False) + '\n')
                        self._update_progress(False)
        
        if self.logger:
            self.logger.info(f"Appended {successful_count}/{len(items)} successful results to {output_file}")
        
        return successful_count


def create_concurrent_processor(model_name: str, enable_logging: bool = True) -> ConcurrentProcessor:
    """
    Factory function to create a concurrent processor for a specific model.
    
    Args:
        model_name: Name of the model
        enable_logging: Whether to enable logging
        
    Returns:
        ConcurrentProcessor instance configured for the model
    """
    return ConcurrentProcessor(model_name=model_name, enable_logging=enable_logging)