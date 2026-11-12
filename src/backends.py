"""Explicit accelerator selection for gprMax; failures never change the solver.

Backend API and prerequisites: https://docs.gprmax.com/en/latest/accelerators.html
Metal uses single precision. Availability only proves device discovery: a model
can still fail compilation or exhaust device memory, and that failure is raised.
"""
import importlib
import platform
import shutil

BACKENDS = ('cpu', 'cuda', 'metal', 'auto')


def probe_backends():
    result = {'cpu': {'available': True, 'device': platform.processor() or platform.machine(), 'reason': 'gprMax CPU/OpenMP'},
              'cuda': {'available': False, 'reason': 'CUDA driver, PyCUDA and nvcc are required'},
              'metal': {'available': False, 'reason': 'Apple Metal requires macOS and PyObjC Metal'}}
    if platform.system() == 'Darwin':
        try:
            metal = importlib.import_module('Metal')
            device = metal.MTLCreateSystemDefaultDevice()
            if device is None:
                result['metal']['reason'] = 'No accessible Metal device in this process (restricted sessions may hide the GPU)'
            else:
                result['metal'] = dict(available=True, device=str(device.name()), reason='Metal device discovered; model compilation checked at run time')
        except (ImportError, RuntimeError, OSError) as exc:
            result['metal']['reason'] = f'Metal bindings unavailable: {exc}'
    if platform.system() != 'Darwin':
        try:
            cuda = importlib.import_module('pycuda.driver'); cuda.init()
            if cuda.Device.count() and shutil.which('nvcc'):
                result['cuda'] = dict(available=True, device=cuda.Device(0).name(), reason='CUDA device 0 and nvcc discovered')
            else:
                result['cuda']['reason'] = 'No CUDA device or nvcc not on PATH'
        except Exception as exc:
            result['cuda']['reason'] = f'CUDA unavailable: {exc}'
    return result


def resolve_backend(requested='cpu'):
    if requested not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}')
    if requested == 'cpu':return 'cpu'
    available = probe_backends()
    if requested == 'auto':
        # This is a capability preference, not a promise of better performance.
        return next((name for name in ('cuda', 'metal') if available[name]['available']), 'cpu')
    if not available[requested]['available']:
        raise ValueError(f'{requested}: {available[requested]["reason"]}')
    return requested


def solver_options(backend):
    if backend == 'cpu':return {}
    if backend == 'cuda':return {'gpu': [0], 'gpu_precision': 'single'}
    if backend == 'metal':return {'metal': True, 'gpu_precision': 'single'}
    raise ValueError('Resolve the backend before constructing solver options')
