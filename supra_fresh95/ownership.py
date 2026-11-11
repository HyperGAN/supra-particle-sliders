"""Retry only pending restorations; never close the caller's runtime owner."""
from copy import deepcopy

class Cleanup:
    def __init__(self, operations):
        self.pending = list(operations)
        self.closing = self.closed = False
    def close(self):
        if self.closed:
            return
        self.closing = True
        failed, errors = [], []
        for operation in self.pending:
            try:
                operation()
            except BaseException as error:
                failed.append(operation)
                errors.append(error)
        self.pending = failed
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup('Supra pending native restoration failed', errors)
        self.closed, self.closing = True, False

def snapshot(runtime):
    """Same native attributes as native_io.snapshot_native, independently pending."""
    from particle_sliders.lifecycle import native_cleanup
    operations = [native_cleanup(root) for root in
        (runtime.model, getattr(runtime, 'text_encoder', None), getattr(runtime, 'vae', None))
        if root is not None]
    cache, branch = runtime.cache, runtime.cfg_adapter_branch
    had_encode, encode = 'encode' in vars(runtime), vars(runtime).get('encode')
    tokens = getattr(runtime, 'token_counts', None)
    token_values = deepcopy(tokens)
    operations += [lambda: setattr(runtime, 'cache', cache),
                   lambda: setattr(runtime, 'cfg_adapter_branch', branch)]
    operations.append((lambda: setattr(runtime, 'encode', encode)) if had_encode
                      else (lambda: vars(runtime).pop('encode', None)))
    if tokens is not None:
        def restore_tokens():
            tokens.clear()
            tokens.update(token_values)
            runtime.token_counts = tokens
        operations.append(restore_tokens)
    operations += [lambda module=module, value=module.multiplier:
                   setattr(module, 'multiplier', value) for module in runtime.loras]
    return Cleanup(operations)

class Owner:
    def __init__(self):
        self.closed = self.closing = False
        self.cleanup = None
    def require_open(self):
        if self.closed or self.closing:
            raise RuntimeError('Supra owner is closed or cleanup is pending')
    def close(self):
        if self.closed:
            return
        self.closing = True
        if self.cleanup is not None:
            self.cleanup.close()
        self.closed, self.closing = True, False
    def __enter__(self):
        self.require_open()
        return self
    def __exit__(self, kind, primary, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup('Supra work and pending cleanup failed', [primary, cleanup])
            raise
