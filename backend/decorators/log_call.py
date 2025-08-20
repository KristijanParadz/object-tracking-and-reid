def log_call(fn):
    """Log when a function is called (just for visibility)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[log] Calling {fn.__qualname__}")
        return fn(*args, **kwargs)
    return wrapper
