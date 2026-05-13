def should_report_progress(batch_idx, total_batches):
    if total_batches <= 0:
        return False
    interval = max(1, total_batches // 10)
    return batch_idx == 1 or batch_idx == total_batches or batch_idx % interval == 0


def print_batch_progress(prefix, batch_idx, total_batches, seen, total):
    total_text = str(total) if total is not None else "?"
    print(
        f"{prefix}: batch {batch_idx}/{total_batches}, images {seen}/{total_text}",
        flush=True,
    )
